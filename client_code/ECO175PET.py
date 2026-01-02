from ._anvil_designer import ECO175PETTemplate
from anvil import *
import plotly.graph_objects as go
import anvil.server

class ECO175PET(ECO175PETTemplate):
  def __init__(self, **properties):
    self.init_components(**properties)
    self._curve_data = [
      {
        "label": "30%",
        "color": "#9aa0a6",
        "points": [
          (0, 90),
          (40, 80),
          (80, 60),
          (120, 40),
          (150, 20),
          (170, 0),
        ],
      },
      {
        "label": "50%",
        "color": "#1a73e8",
        "points": [
          (0, 150),
          (50, 130),
          (100, 110),
          (140, 80),
          (170, 50),
          (200, 20),
          (210, 0),
        ],
      },
      {
        "label": "75%",
        "color": "#fbbc04",
        "points": [
          (0, 220),
          (50, 200),
          (100, 170),
          (150, 130),
          (180, 90),
          (210, 40),
          (230, 0),
        ],
      },
      {
        "label": "100%",
        "color": "#34a853",
        "points": [
          (0, 300),
          (50, 280),
          (100, 240),
          (150, 190),
          (200, 130),
          (230, 70),
          (250, 0),
        ],
      },
    ]
    self._selected_point = None
    self._render_curve_plot()

  def _get_curve_points(self, label):
    for curve in self._curve_data:
      if curve["label"] == label:
        return curve["points"]
    return []

  def _interpolate_curve_pressure(self, flow, points):
    if not points:
      return None
    points = sorted(points, key=lambda p: p[0])
    min_flow = points[0][0]
    max_flow = points[-1][0]
    if flow < min_flow or flow > max_flow:
      return None
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
      if x0 <= flow <= x1:
        if x1 == x0:
          return y0
        ratio = (flow - x0) / (x1 - x0)
        return y0 + ratio * (y1 - y0)
    return None

  def _build_clickable_points(self, max_curve_points, flow_step=5, pressure_step=5):
    max_curve_points = sorted(max_curve_points, key=lambda p: p[0])
    min_flow = int(max_curve_points[0][0])
    max_flow = int(max_curve_points[-1][0])
    flows = []
    pressures = []
    for flow in range(min_flow, max_flow + 1, flow_step):
      max_pressure = self._interpolate_curve_pressure(flow, max_curve_points)
      if max_pressure is None:
        continue
      max_pressure = int(max_pressure)
      for pressure in range(0, max_pressure + 1, pressure_step):
        flows.append(flow)
        pressures.append(pressure)
    return flows, pressures

  def _render_curve_plot(self):
    fig = go.Figure()
    for curve in self._curve_data:
      flows = [p[0] for p in curve["points"]]
      pressures = [p[1] for p in curve["points"]]
      fig.add_trace(
        go.Scatter(
          x=flows,
          y=pressures,
          mode="lines",
          name=curve["label"],
          line={"color": curve["color"], "width": 2},
        )
      )

    max_curve_points = self._get_curve_points("100%")
    if max_curve_points:
      click_flows, click_pressures = self._build_clickable_points(max_curve_points)
      fig.add_trace(
        go.Scatter(
          x=click_flows,
          y=click_pressures,
          mode="markers",
          marker={"size": 6, "opacity": 0.01, "color": "#000000"},
          name="Vælg driftspunkt",
          showlegend=False,
          hoverinfo="skip",
        )
      )

    if self._selected_point:
      fig.add_trace(
        go.Scatter(
          x=[self._selected_point[0]],
          y=[self._selected_point[1]],
          mode="markers",
          marker={"size": 10, "color": "#d93025"},
          name="Valgt punkt",
        )
      )

    fig.update_layout(
      margin={"l": 40, "r": 20, "t": 30, "b": 40},
      xaxis={"title": "Luftmængde (m3/h)"},
      yaxis={"title": "Tryk (Pa)", "rangemode": "tozero"},
      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
      clickmode="event+select",
    )
    self.plot_operating_point.figure = fig

  def plot_operating_point_click(self, **event_args):
    points = event_args.get("points") or []
    if not points:
      return
    point = points[0]
    try:
      flow = float(point["x"])
      pressure = float(point["y"])
    except (KeyError, TypeError, ValueError):
      return

    max_curve_points = self._get_curve_points("100%")
    max_pressure = self._interpolate_curve_pressure(flow, max_curve_points)
    if max_pressure is None or pressure > max_pressure:
      Notification("Vælg et punkt indenfor 100% kurven.", style="warning").show()
      return

    self.flow_input.text = f"{round(flow)}"
    self.pressure_input.text = f"{round(pressure)}"
    self._selected_point = (flow, pressure)
    self._render_curve_plot()

  def categorize_and_render(self, result_dict):
    def pick(cat_keys):
      picked = {k: v for k, v in result_dict.items() if any(kk.lower() in str(k).lower() for kk in cat_keys)}
      return picked
    sfp = pick(['sfp','sel','fan'])
    sound = pick(['lw','lp','sound','lyd'])
    heat = {k: v for k, v in result_dict.items() if k not in sfp and k not in sound}
    def render(d):
      if not d:
        return "—"
      return "\n".join(f"{k}: {v}" for k, v in d.items())
    if hasattr(self, 'out_sfp'): self.out_sfp.content = render(sfp)
    if hasattr(self, 'out_sound'): self.out_sound.content = render(sound)
    if hasattr(self, 'out_heat'): self.out_heat.content = render(heat)

  def calc_button_click(self, **event_args):
    try:
      flow = float(self.flow_input.text)
      temp = float(self.temp_input.text)
      pressure = float(self.pressure_input.text)
      result = anvil.server.call('run_ECO175PET', flow, temp, pressure)
      self.categorize_and_render(result)
    except Exception as e:
      msg = f"Fejl: {e}"
      if hasattr(self, 'out_sfp'): self.out_sfp.content = msg
      if hasattr(self, 'out_sound'): self.out_sound.content = msg
      if hasattr(self, 'out_heat'): self.out_heat.content = msg
