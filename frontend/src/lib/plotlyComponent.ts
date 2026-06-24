import createPlotlyComponent from "react-plotly.js/factory";
// plotly 번들 엔트리 (react-plotly와 동일 경로)
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error dist/plotly has no separate typings
import Plotly from "plotly.js/dist/plotly";

/** react-plotly와 동일한 Plotly 번들을 공유하는 Plot 컴포넌트 */
export { Plotly };
export default createPlotlyComponent(Plotly);
