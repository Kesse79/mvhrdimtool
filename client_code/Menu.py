from ._anvil_designer import MenuTemplate
from anvil import *

PRODUCTS = ["ECO400XL", "ECO360", "ECO375", "ECO190XL", "ECO175PET"]

class Menu(MenuTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Populate the dropdown with (label,value) tuples so the displayed text
    # and the value we read are identical.
    self.product_dropdown.items = [(p, p) for p in PRODUCTS]

  def open_button_click(self, **event_args):
    """Triggered when the 'Open' button is pressed"""
    self._open_selected_form()

  def product_dropdown_change(self, **event_args):
    """Triggered immediately after the user picks a product in the dropdown"""
    self._open_selected_form()

  # -------------------------
  #  HELPER METHODS
  # -------------------------
  def _open_selected_form(self):
    sel = self.product_dropdown.selected_value
    if sel:
      # open_form accepts either a form class, an instance, or the
      # dotted‑path string of the form class. Because our form classes
      # are top‑level and named exactly like the product codes, we can
      # simply pass the string.
      open_form(sel)
