import tkinter as tk
from gui import App

if __name__ == "__main__":
    window = tk.Tk()
    window.title("Detección de Objetos Prohibidos en Banco")
    app = App(window)
    window.protocol("WM_DELETE_WINDOW", app.on_closing)
    window.mainloop()
    