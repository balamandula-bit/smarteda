
SmartEDA — AI-Driven VLSI Layout Generator
==========================================

HOW TO RUN (3 steps)
---------------------

Step 1 — Install requirements
Open terminal / VS Code terminal and run:
    pip install flask matplotlib numpy

Step 2 — Run the app
    python app.py

Step 3 — Open your browser and go to:
    http://127.0.0.1:5000

That's it! The app will open in your browser.

HOW TO USE
----------
1. Add cells using the left panel (name, type, width, height)
2. Add connections between cells
3. Click "Generate VLSI Layout"
4. See the AI-optimized layout with PPA metrics
5. Download the PNG image

OR just click one of the preset buttons (8-bit ALU, Counter, MUX)
to load a ready-made circuit and generate instantly.

CELL TYPES AVAILABLE
--------------------
logic, register, mux, flipflop, gate, memory, io, buffer

FREE TOOLS USED
---------------
- Python
- Flask (web framework)
- NumPy (math)
- Matplotlib (drawing)
- Simulated Annealing (AI optimizer)
