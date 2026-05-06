import { h, render } from "preact";
import { App } from "./app";
import { ViewerApp } from "./viewer_app";

const mode = new URLSearchParams(window.location.search).get("mode");
const Root = mode === "viewer" ? ViewerApp : App;

render(<Root />, document.getElementById("app")!);
