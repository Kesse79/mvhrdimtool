from ._anvil_designer import ECO400XLTemplate
from anvil import *
import anvil.server

class ECO400XL(ECO400XLTemplate):
  def __init__(self, **properties):
    self.init_components(**properties)

  def categorize_and_render(self, result_dict):
    def pick(cat_keys):
      picked = {k: v for k, v in result_dict.items() if any(kk.lower() in str(k).lower() for kk in cat_keys)}
      return picked
    sfp = pick(['sfp', 'sel', 'fan', 'ventilator'])
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
      result = anvil.server.call('run_ECO300XL', flow, temp, pressure)
      self.categorize_and_render(result)
    except Exception as e:
      msg = f"Fejl: {e}"
      if hasattr(self, 'out_sfp'): self.out_sfp.content = msg
      if hasattr(self, 'out_sound'): self.out_sound.content = msg
      if hasattr(self, 'out_heat'): self.out_heat.content = msg

  def nav_link_click(self, **event_args):
    sender = event_args.get('sender')
    if sender and sender.tag:
      open_form(sender.tag)
