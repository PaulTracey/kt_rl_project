# help_ui.py
# --------------------------------------------------------------
# help UI module
#
# Defines the HelpUI class for the Tactical Advisor Prototype (TAP).
#
# used for:
# WIP. Provides a help panel, accessible via a "Help ?" button.
# --------------------------------------------------------------
import pygame_gui
import pygame



class HelpUI:
    def __init__(self, ui_manager, container, window_width, window_height, panel_width=250):
        self.manager = ui_manager
        self.help_visible = False

        # Create Help Button
        self.help_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 780), (panel_width - 40, 30)),
            text="Help ?",
            container=container,
            manager=ui_manager
        )

        # Create Help Panel (Initially hidden)
        self.help_panel = pygame_gui.elements.UITextBox(
            html_text=self._help_text(),
            relative_rect=pygame.Rect(
                (window_width // 2 - 200, window_height // 2 - 150),
                (400, 300)
            ),
            manager=ui_manager,
            visible=False
        )


    def _help_text(self):
        # The text shown inside the help pop-up
        return (
           "<b>Welcome to Tap</b>"
           "<br>---------------"
           "<br>Controls:<br>"
           "– Left-click to select a unit<br>"
           "– Right-click to move or attack<br>"
           "– Press <b>P</b> to pass a turn<br>"
           "– Use dropdowns to deploy teams<br>"
           "<br>"
           "First to eliminate the enemy or score objectives wins!"

        )

    def process_event(self, event):
        # Toggle the help panel when the Help button is pressed
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.help_button:
                self.help_visible = not self.help_visible
                self.help_panel.visible = self.help_visible
