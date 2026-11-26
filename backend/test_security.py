import asyncio
import json
import os
import shlex
import socket
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import requests
from fastapi import HTTPException

from backend import auth, security, server, session_allocator


PUBLIC_DNS = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]


class SecurityTests(unittest.TestCase):
    def test_url_policy(self):
        with patch.dict(os.environ, {'WI2_ALLOWED_HTTP_HOSTS': 'storage.example.org'}), patch('socket.getaddrinfo', return_value=PUBLIC_DNS):
            security.validate_url('https://storage.example.org/image.dzip')
            for url in ('http://data-proxy.ebrains.eu/x', 'https://data-proxy.ebrains.eu.evil.org/x', 'https://evil.org/data-proxy.ebrains.eu', 'https://user@data-proxy.ebrains.eu/x', 'https://data-proxy.ebrains.eu:444/x', 'https://127.0.0.1/x'):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    security.validate_url(url)
            with self.assertRaises(ValueError):
                security.validate_url('https://storage.example.org/x', proxy_only=True)
        with patch('socket.getaddrinfo', return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))]), self.assertRaises(ValueError):
            security.validate_url('https://data-proxy.ebrains.eu/x')

    def test_credentials_only_sent_to_proxy_and_redirects_blocked(self):
        response = requests.Response()
        response.status_code = 200
        with patch('backend.security.validate_url'), patch.object(requests.Session, 'request', return_value=response) as send:
            session = security.Session()
            session.headers['Authorization'] = 'Bearer secret'
            session.get('https://storage.example.org/x')
            self.assertIsNone(send.call_args.kwargs['headers']['Authorization'])
            self.assertFalse(send.call_args.kwargs['allow_redirects'])
            session.get('https://data-proxy.ebrains.eu/x')
            self.assertEqual(send.call_args.kwargs['headers']['Authorization'], 'Bearer secret')
            response.status_code = 302
            response._content = b''
            response._content_consumed = True
            with self.assertRaises(ValueError):
                session.get('https://data-proxy.ebrains.eu/x')

    def test_workspace_identity_and_output_containment(self):
        a = security.workspace_name({'sub': '1', 'preferred_username': '../../alice'})
        b = security.workspace_name({'sub': '2', 'preferred_username': '../../alice'})
        self.assertNotEqual(a, b)
        self.assertNotIn('/', a)
        self.assertIn('alice', a)
        for name in ('../../escape', '/tmp/escape', 'C:\\escape', '..', 'folder/image'):
            output = security.safe_output_name(name)
            root = Path('test-root').resolve()
            self.assertTrue((root / (output + '.dzi')).resolve().is_relative_to(root))

    def test_data_proxy_session_retains_url_policy(self):
        session = auth.DataProxySession()
        session.headers['Authorization'] = 'Bearer secret'
        response = requests.Response()
        response.status_code = 200
        with patch('backend.security.validate_url') as validate, patch.object(requests.Session, 'request', return_value=response) as send:
            session.get('https://storage.example.org/x')
            validate.assert_called_once_with('https://storage.example.org/x', issued_by_proxy=False)
            self.assertIsNone(send.call_args.kwargs['headers']['Authorization'])
            self.assertFalse(send.call_args.kwargs['allow_redirects'])
        self.assertFalse(session.trust_env)

    def test_job_submission_preserves_workspace_and_provenance(self):
        identity = {'sub': "user'123", 'preferred_username': '../../alice'}
        source = 'https://data-proxy.ebrains.eu/api/v1/buckets/test/image.dzip'
        req = session_allocator.HeadlessJobRequest(
            annotations=[{'dzip_url': source, 'strokes': []}],
            p_source=source, output_dir=source + '/out', cpus=4,
        )
        ssh = AsyncMock(side_effect=['', '12345;cluster'])
        with patch.object(session_allocator, '_identity', return_value=identity), patch.object(session_allocator, 'validate_url'), patch.object(session_allocator, '_ssh_run', ssh):
            job = asyncio.run(session_allocator.create_headless_job(req, 'Bearer secret'))
        try:
            workspace = security.workspace_name(identity)
            self.assertIn('/wi2_users/' + workspace + '/logs/', job['log_path'])
            self.assertEqual(job['training_sources'], [source])
            script = ssh.call_args_list[1].kwargs['stdin']
            line = next(line for line in script.splitlines() if line.startswith('printf ') and 'training_sources' in line)
            provenance = json.loads(shlex.split(line)[2])
            self.assertEqual(provenance['user_id'], identity['sub'])
            self.assertEqual(provenance['training_sources'], [source])
        finally:
            session_allocator._jobs.pop(job['job_id'], None)

    def test_export_ownership(self):
        for registry, endpoint in ((server._exports, server.export_status), (server._batch_jobs, server.batch_export_status)):
            registry['test-job'] = {'user_id': 'alice', 'status': 'done'}
            try:
                with patch.object(server, '_auth', return_value={'sub': 'alice'}):
                    self.assertEqual(asyncio.run(endpoint('test-job', 'Bearer x'))['status'], 'done')
                with patch.object(server, '_auth', return_value={'sub': 'bob'}), self.assertRaises(HTTPException) as error:
                    asyncio.run(endpoint('test-job', 'Bearer x'))
                self.assertEqual(error.exception.status_code, 404)
            finally:
                registry.pop('test-job', None)

    def test_save_project_auth_before_network(self):
        req = server.SaveProjectRequest(json_content='{}', dest_url='https://evil.org/x')
        with patch.object(server, '_auth', side_effect=HTTPException(401, 'invalid')), patch.object(server, '_dp_put') as upload, self.assertRaises(HTTPException):
            asyncio.run(server.save_project_to_proxy(req, 'Bearer fake'))
        upload.assert_not_called()

    def test_shell_arguments_and_username_log_paths(self):
        malicious = "https://example.org/a' $(touch /tmp/owned); echo '"
        script = session_allocator._build_sbatch_script(job_id='alice-job1', annotations_b64='W10=', t_source=malicious, p_source=malicious, output_dir=malicious, features_json="{\"x\":\"a'b\"}", level=2, token="a'b", cpus=4, cache_namespace='safe/source', workspace='alice-safe')
        for line in script.splitlines():
            if '--p-source ' in line or '--output-dir ' in line or '--t-source ' in line:
                self.assertEqual(shlex.split(line.rstrip(' \\'))[1], malicious)
            if line.startswith('printf '):
                self.assertIn(malicious, shlex.split(line)[2])
        self.assertIn('/wi2_users/alice-safe/logs/wi2-run-%j.log', script)
        self.assertIn('-oStrictHostKeyChecking=yes', session_allocator._ssh_cmd('true'))


if __name__ == '__main__':
    unittest.main()
