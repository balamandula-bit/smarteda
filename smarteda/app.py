"""
SmartEDA Backend — Extended Version
=====================================
AI Algorithms: Simulated Annealing, Genetic Algorithm, Particle Swarm Optimization
Metrics: Wire Length, Power, Timing, Area Utilization
Presets: 10 circuits
Export: PNG, CSV
"""

import csv
import base64
import io
import time
import math
import random
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

CELL_COLORS = {
    "logic": "#7c3aed", "register": "#0891b2", "mux": "#d97706",
    "flipflop": "#059669", "gate": "#dc2626", "memory": "#1d4ed8",
    "io": "#047857", "buffer": "#b45309",
}
CELL_TYPES = list(CELL_COLORS.keys())

PRESETS = {
    "basic_gates": {"name": "Basic Logic Gates", "cells": [
        {"name": "IN1", "type": "io", "w": 2, "h": 2}, {
            "name": "IN2", "type": "io", "w": 2, "h": 2},
        {"name": "AND1", "type": "gate", "w": 1, "h": 1}, {
            "name": "OR1", "type": "gate", "w": 1, "h": 1},
        {"name": "NOT1", "type": "gate", "w": 1, "h": 1}, {
            "name": "NAND1", "type": "gate", "w": 1, "h": 1},
        {"name": "NOR1", "type": "gate", "w": 1, "h": 1}, {
            "name": "OUT", "type": "io", "w": 2, "h": 2},
    ], "connections": [
        ["IN1", "AND1"], ["IN2", "AND1"], ["IN1", "OR1"], ["IN2", "OR1"],
        ["AND1", "NOT1"], ["OR1", "NAND1"], ["NOT1", "NOR1"], [
            "NAND1", "NOR1"], ["NOR1", "OUT"],
    ]},
    "adder": {"name": "4-bit Adder", "cells": [
        {"name": "A0", "type": "io", "w": 2, "h": 1}, {
            "name": "A1", "type": "io", "w": 2, "h": 1},
        {"name": "A2", "type": "io", "w": 2, "h": 1}, {
            "name": "A3", "type": "io", "w": 2, "h": 1},
        {"name": "B0", "type": "io", "w": 2, "h": 1}, {
            "name": "B1", "type": "io", "w": 2, "h": 1},
        {"name": "B2", "type": "io", "w": 2, "h": 1}, {
            "name": "B3", "type": "io", "w": 2, "h": 1},
        {"name": "FA0", "type": "logic", "w": 3, "h": 2}, {
            "name": "FA1", "type": "logic", "w": 3, "h": 2},
        {"name": "FA2", "type": "logic", "w": 3, "h": 2}, {
            "name": "FA3", "type": "logic", "w": 3, "h": 2},
        {"name": "S0", "type": "io", "w": 2, "h": 1}, {
            "name": "S1", "type": "io", "w": 2, "h": 1},
        {"name": "S2", "type": "io", "w": 2, "h": 1}, {
            "name": "S3", "type": "io", "w": 2, "h": 1},
        {"name": "COUT", "type": "io", "w": 2, "h": 1},
    ], "connections": [
        ["A0", "FA0"], ["B0", "FA0"], ["FA0", "S0"],
        ["A1", "FA1"], ["B1", "FA1"], ["FA0", "FA1"], ["FA1", "S1"],
        ["A2", "FA2"], ["B2", "FA2"], ["FA1", "FA2"], ["FA2", "S2"],
        ["A3", "FA3"], ["B3", "FA3"], ["FA2", "FA3"], [
            "FA3", "S3"], ["FA3", "COUT"],
    ]},
    "counter": {"name": "4-bit Counter", "cells": [
        {"name": "FF0", "type": "flipflop", "w": 2, "h": 1}, {
            "name": "FF1", "type": "flipflop", "w": 2, "h": 1},
        {"name": "FF2", "type": "flipflop", "w": 2, "h": 1}, {
            "name": "FF3", "type": "flipflop", "w": 2, "h": 1},
        {"name": "AND1", "type": "gate", "w": 1, "h": 1}, {
            "name": "AND2", "type": "gate", "w": 1, "h": 1},
        {"name": "OR1", "type": "gate", "w": 1, "h": 1},
        {"name": "CLK", "type": "io", "w": 2, "h": 2}, {
            "name": "OUT", "type": "io", "w": 2, "h": 2},
    ], "connections": [
        ["CLK", "FF0"], ["FF0", "FF1"], ["FF1", "FF2"], ["FF2", "FF3"],
        ["FF3", "AND1"], ["AND1", "AND2"], ["AND2", "OR1"], ["OR1", "OUT"],
        ["FF0", "AND1"], ["FF1", "AND2"],
    ]},
    "mux": {"name": "4:1 Multiplexer", "cells": [
        {"name": "IN0", "type": "io", "w": 2, "h": 2}, {
            "name": "IN1", "type": "io", "w": 2, "h": 2},
        {"name": "IN2", "type": "io", "w": 2, "h": 2}, {
            "name": "IN3", "type": "io", "w": 2, "h": 2},
        {"name": "SEL0", "type": "io", "w": 2, "h": 1}, {
            "name": "SEL1", "type": "io", "w": 2, "h": 1},
        {"name": "NOT1", "type": "gate", "w": 1, "h": 1}, {
            "name": "NOT2", "type": "gate", "w": 1, "h": 1},
        {"name": "AND1", "type": "gate", "w": 1, "h": 1}, {
            "name": "AND2", "type": "gate", "w": 1, "h": 1},
        {"name": "AND3", "type": "gate", "w": 1, "h": 1}, {
            "name": "AND4", "type": "gate", "w": 1, "h": 1},
        {"name": "OR1", "type": "gate", "w": 2, "h": 2}, {
            "name": "OUT", "type": "io", "w": 2, "h": 2},
    ], "connections": [
        ["IN0", "AND1"], ["IN1", "AND2"], ["IN2", "AND3"], ["IN3", "AND4"],
        ["SEL0", "NOT1"], ["SEL1", "NOT2"], ["NOT1", "AND1"], ["NOT2", "AND2"],
        ["SEL0", "AND3"], ["SEL1", "AND4"],
        ["AND1", "OR1"], ["AND2", "OR1"], [
            "AND3", "OR1"], ["AND4", "OR1"], ["OR1", "OUT"],
    ]},
    "alu": {"name": "8-bit ALU", "cells": [
        {"name": "ALU", "type": "logic", "w": 4, "h": 3},
        {"name": "REG_A", "type": "register", "w": 2, "h": 2}, {
            "name": "REG_B", "type": "register", "w": 2, "h": 2},
        {"name": "MUX1", "type": "mux", "w": 2, "h": 2}, {
            "name": "MUX2", "type": "mux", "w": 2, "h": 2},
        {"name": "FF1", "type": "flipflop", "w": 2, "h": 1}, {
            "name": "FF2", "type": "flipflop", "w": 2, "h": 1},
        {"name": "AND1", "type": "gate", "w": 1, "h": 1}, {
            "name": "OR1", "type": "gate", "w": 1, "h": 1},
        {"name": "RAM", "type": "memory", "w": 4, "h": 4},
        {"name": "IO_IN", "type": "io", "w": 2, "h": 3}, {
            "name": "IO_OUT", "type": "io", "w": 2, "h": 3},
    ], "connections": [
        ["IO_IN", "REG_A"], ["IO_IN", "REG_B"], [
            "REG_A", "MUX1"], ["REG_B", "MUX2"],
        ["MUX1", "ALU"], ["MUX2", "ALU"], ["ALU", "FF1"], ["ALU", "FF2"],
        ["FF1", "RAM"], ["FF2", "RAM"], ["RAM", "IO_OUT"],
        ["ALU", "AND1"], ["AND1", "OR1"], ["OR1", "IO_OUT"],
    ]},
    "register_file": {"name": "Register File", "cells": [
        {"name": "REG0", "type": "register", "w": 3, "h": 2}, {
            "name": "REG1", "type": "register", "w": 3, "h": 2},
        {"name": "REG2", "type": "register", "w": 3, "h": 2}, {
            "name": "REG3", "type": "register", "w": 3, "h": 2},
        {"name": "MUX_A", "type": "mux", "w": 2, "h": 2}, {
            "name": "MUX_B", "type": "mux", "w": 2, "h": 2},
        {"name": "DEC", "type": "logic", "w": 2, "h": 2}, {
            "name": "WE", "type": "gate", "w": 1, "h": 1},
        {"name": "BUS_A", "type": "io", "w": 2, "h": 3}, {
            "name": "BUS_B", "type": "io", "w": 2, "h": 3},
        {"name": "DATA_IN", "type": "io", "w": 2, "h": 2}, {
            "name": "ADDR", "type": "io", "w": 2, "h": 2},
    ], "connections": [
        ["ADDR", "DEC"], ["DEC", "REG0"], ["DEC", "REG1"], [
            "DEC", "REG2"], ["DEC", "REG3"],
        ["DATA_IN", "WE"], ["WE", "REG0"], ["WE", "REG1"],
        ["REG0", "MUX_A"], ["REG1", "MUX_A"], [
            "REG2", "MUX_B"], ["REG3", "MUX_B"],
        ["MUX_A", "BUS_A"], ["MUX_B", "BUS_B"],
    ]},
    "uart": {"name": "UART Controller", "cells": [
        {"name": "TX_REG", "type": "register", "w": 3, "h": 2}, {
            "name": "RX_REG", "type": "register", "w": 3, "h": 2},
        {"name": "BAUD", "type": "logic", "w": 3, "h": 2},
        {"name": "TX_FSM", "type": "logic", "w": 3, "h": 3}, {
            "name": "RX_FSM", "type": "logic", "w": 3, "h": 3},
        {"name": "FIFO_TX", "type": "memory", "w": 3, "h": 3}, {
            "name": "FIFO_RX", "type": "memory", "w": 3, "h": 3},
        {"name": "FF1", "type": "flipflop", "w": 2, "h": 1}, {
            "name": "FF2", "type": "flipflop", "w": 2, "h": 1},
        {"name": "MUX1", "type": "mux", "w": 2, "h": 2},
        {"name": "TX_PIN", "type": "io", "w": 2, "h": 2}, {
            "name": "RX_PIN", "type": "io", "w": 2, "h": 2},
        {"name": "CLK", "type": "io", "w": 2, "h": 2}, {
            "name": "DATA_BUS", "type": "io", "w": 3, "h": 2},
    ], "connections": [
        ["CLK", "BAUD"], ["BAUD", "TX_FSM"], ["BAUD", "RX_FSM"],
        ["DATA_BUS", "TX_REG"], ["TX_REG", "FIFO_TX"], ["FIFO_TX", "TX_FSM"],
        ["TX_FSM", "FF1"], ["FF1", "MUX1"], ["MUX1", "TX_PIN"],
        ["RX_PIN", "FF2"], ["FF2", "RX_FSM"], ["RX_FSM", "FIFO_RX"],
        ["FIFO_RX", "RX_REG"], ["RX_REG", "DATA_BUS"],
    ]},
    "vga": {"name": "VGA Controller", "cells": [
        {"name": "H_SYNC", "type": "logic", "w": 3, "h": 2}, {
            "name": "V_SYNC", "type": "logic", "w": 3, "h": 2},
        {"name": "PIX_CLK", "type": "logic", "w": 2, "h": 2},
        {"name": "FRAME_BUF", "type": "memory", "w": 5, "h": 4},
        {"name": "COLOR_MUX", "type": "mux", "w": 3, "h": 2},
        {"name": "R_DAC", "type": "logic", "w": 2, "h": 2}, {
            "name": "G_DAC", "type": "logic", "w": 2, "h": 2},
        {"name": "B_DAC", "type": "logic", "w": 2, "h": 2},
        {"name": "FF1", "type": "flipflop", "w": 2, "h": 1}, {
            "name": "FF2", "type": "flipflop", "w": 2, "h": 1},
        {"name": "CLK_IN", "type": "io", "w": 2, "h": 2}, {
            "name": "VGA_OUT", "type": "io", "w": 3, "h": 3},
        {"name": "MEM_BUS", "type": "io", "w": 3, "h": 2},
    ], "connections": [
        ["CLK_IN", "PIX_CLK"], ["PIX_CLK", "H_SYNC"], ["PIX_CLK", "V_SYNC"],
        ["H_SYNC", "FF1"], ["V_SYNC", "FF2"],
        ["MEM_BUS", "FRAME_BUF"], ["FRAME_BUF", "COLOR_MUX"],
        ["COLOR_MUX", "R_DAC"], ["COLOR_MUX", "G_DAC"], ["COLOR_MUX", "B_DAC"],
        ["R_DAC", "VGA_OUT"], ["G_DAC", "VGA_OUT"], ["B_DAC", "VGA_OUT"],
        ["FF1", "COLOR_MUX"], ["FF2", "COLOR_MUX"],
    ]},
    "cpu": {"name": "16-bit CPU Core", "cells": [
        {"name": "PC", "type": "register", "w": 3, "h": 2}, {
            "name": "IR", "type": "register", "w": 3, "h": 2},
        {"name": "MAR", "type": "register", "w": 2, "h": 2}, {
            "name": "MDR", "type": "register", "w": 2, "h": 2},
        {"name": "ACC", "type": "register", "w": 3, "h": 2},
        {"name": "ALU", "type": "logic", "w": 5, "h": 4}, {
            "name": "CU", "type": "logic", "w": 4, "h": 3},
        {"name": "RAM", "type": "memory", "w": 5, "h": 5}, {
            "name": "ROM", "type": "memory", "w": 4, "h": 4},
        {"name": "MUX1", "type": "mux", "w": 2, "h": 2}, {
            "name": "MUX2", "type": "mux", "w": 2, "h": 2},
        {"name": "MUX3", "type": "mux", "w": 2, "h": 2},
        {"name": "FF1", "type": "flipflop", "w": 2, "h": 1}, {
            "name": "FF2", "type": "flipflop", "w": 2, "h": 1},
        {"name": "FF3", "type": "flipflop", "w": 2, "h": 1}, {
            "name": "FF4", "type": "flipflop", "w": 2, "h": 1},
        {"name": "AND1", "type": "gate", "w": 1, "h": 1}, {
            "name": "AND2", "type": "gate", "w": 1, "h": 1},
        {"name": "OR1", "type": "gate", "w": 1, "h": 1}, {
            "name": "NOT1", "type": "gate", "w": 1, "h": 1},
        {"name": "BUF1", "type": "buffer", "w": 1, "h": 1}, {
            "name": "BUF2", "type": "buffer", "w": 1, "h": 1},
        {"name": "IO_IN", "type": "io", "w": 2, "h": 3}, {
            "name": "IO_OUT", "type": "io", "w": 2, "h": 3},
    ], "connections": [
        ["IO_IN", "PC"], ["IO_IN", "IR"], ["PC", "MAR"], ["IR", "CU"],
        ["MAR", "RAM"], ["MDR", "RAM"], ["RAM", "MDR"], ["MDR", "ACC"],
        ["ACC", "ALU"], ["ALU", "ACC"], ["ALU", "FF1"], ["ALU", "FF2"],
        ["CU", "MUX1"], ["CU", "MUX2"], ["CU", "MUX3"],
        ["MUX1", "ALU"], ["MUX2", "ALU"], ["MUX3", "MAR"],
        ["FF1", "AND1"], ["FF2", "AND2"], ["AND1", "OR1"], ["AND2", "OR1"],
        ["OR1", "FF3"], ["FF3", "BUF1"], ["BUF1", "IO_OUT"],
        ["NOT1", "AND1"], ["ALU", "NOT1"], ["ROM", "CU"], ["PC", "ROM"],
        ["FF4", "BUF2"], ["BUF2", "IO_OUT"], ["OR1", "FF4"],
    ]},
    "soc": {"name": "Simple SoC", "cells": [
        {"name": "CPU", "type": "logic", "w": 5, "h": 4}, {
            "name": "GPU", "type": "logic", "w": 5, "h": 4},
        {"name": "RAM", "type": "memory", "w": 5, "h": 5}, {
            "name": "ROM", "type": "memory", "w": 4, "h": 4},
        {"name": "DMA", "type": "logic", "w": 3, "h": 3}, {
            "name": "BUS", "type": "logic", "w": 6, "h": 2},
        {"name": "UART", "type": "logic", "w": 3, "h": 2}, {
            "name": "SPI", "type": "logic", "w": 3, "h": 2},
        {"name": "I2C", "type": "logic", "w": 3, "h": 2}, {
            "name": "GPIO", "type": "io", "w": 3, "h": 3},
        {"name": "CLK_GEN", "type": "logic", "w": 3, "h": 2}, {
            "name": "PWR_MGT", "type": "logic", "w": 3, "h": 2},
        {"name": "INT_CTRL", "type": "logic", "w": 3, "h": 2}, {
            "name": "CACHE", "type": "memory", "w": 4, "h": 3},
        {"name": "MMU", "type": "logic", "w": 3, "h": 2},
        {"name": "FF1", "type": "flipflop", "w": 2, "h": 1}, {
            "name": "FF2", "type": "flipflop", "w": 2, "h": 1},
        {"name": "MUX1", "type": "mux", "w": 2, "h": 2}, {
            "name": "MUX2", "type": "mux", "w": 2, "h": 2},
        {"name": "IO_PORT", "type": "io", "w": 3, "h": 4},
    ], "connections": [
        ["CPU", "BUS"], ["GPU", "BUS"], ["DMA", "BUS"],
        ["BUS", "RAM"], ["BUS", "ROM"], ["BUS", "CACHE"],
        ["CPU", "CACHE"], ["CACHE", "RAM"], ["CPU", "MMU"], ["MMU", "RAM"],
        ["BUS", "UART"], ["BUS", "SPI"], ["BUS", "I2C"], ["BUS", "GPIO"],
        ["CLK_GEN", "CPU"], ["CLK_GEN", "GPU"], ["CLK_GEN", "DMA"],
        ["PWR_MGT", "CPU"], ["PWR_MGT", "GPU"],
        ["INT_CTRL", "CPU"], ["UART", "INT_CTRL"], ["GPIO", "INT_CTRL"],
        ["FF1", "MUX1"], ["FF2", "MUX2"], ["MUX1", "BUS"], ["MUX2", "BUS"],
        ["GPIO", "IO_PORT"], ["UART", "IO_PORT"],
    ]},
}

# ── Core helpers ──────────────────────────────────────────────────────


def build_nets(cells, connections):
    idx = {c['name']: i for i, c in enumerate(cells)}
    return [(idx[a], idx[b]) for a, b in connections if a in idx and b in idx]


def random_placement(cells, gw, gh):
    positions = []
    for c in cells:
        w, h = c['w'], c['h']
        placed = False
        for _ in range(500):
            x = random.randint(0, max(0, gw-w))
            y = random.randint(0, max(0, gh-h))
            if all(not (x < ox+ow and x+w > ox and y < oy+oh and y+h > oy) for ox, oy, ow, oh in positions):
                positions.append((x, y, w, h))
                placed = True
                break
        if not placed:
            positions.append((random.randint(0, max(0, gw-w)),
                             random.randint(0, max(0, gh-h)), w, h))
    return positions


def wirelength(pos, nets):
    total = 0
    for i, j in nets:
        if i >= len(pos) or j >= len(pos):
            continue
        x1, y1, w1, h1 = pos[i]
        x2, y2, w2, h2 = pos[j]
        total += abs(x1+w1/2-x2-w2/2)+abs(y1+h1/2-y2-h2/2)
    return total


def count_overlaps(pos):
    count = 0
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            x1, y1, w1, h1 = pos[i]
            x2, y2, w2, h2 = pos[j]
            if x1 < x2+w2 and x1+w1 > x2 and y1 < y2+h2 and y1+h1 > y2:
                count += 1
    return count


def area_utilization(pos, gw, gh):
    return round(sum(w*h for x, y, w, h in pos)/(gw*gh)*100, 1)


def estimate_power(wl, cells):
    base = wl * 0.12
    for c in cells:
        base += {"memory": 15, "logic": 8, "flipflop": 3}.get(c['type'], 2)
    return round(base, 1)


def estimate_timing(wl, nets):
    avg = wl/max(len(nets), 1)
    return round(avg*0.05+len(nets)*0.1, 2)


def cost(pos, nets, gw, gh):
    return wirelength(pos, nets) + count_overlaps(pos)*60 + sum(100 for x, y, w, h in pos if x < 0 or y < 0 or x+w > gw or y+h > gh)

# ── AI Algorithm 1: Simulated Annealing ──────────────────────────────


def simulated_annealing(cells, nets, gw, gh, iters=10000):
    t0 = time.time()
    cur = random_placement(cells, gw, gh)
    cur_cost = cost(cur, nets, gw, gh)
    best = list(cur)
    best_cost = cur_cost
    T = 120.0
    for _ in range(iters):
        T = max(0.1, T*0.9995)
        i = random.randint(0, len(cells)-1)
        x, y, w, h = cur[i]
        npos = list(cur)
        npos[i] = (random.randint(0, max(0, gw-w)),
                   random.randint(0, max(0, gh-h)), w, h)
        nc = cost(npos, nets, gw, gh)
        d = nc-cur_cost
        if d < 0 or (T > 0 and random.random() < math.exp(-d/T)):
            cur = npos
            cur_cost = nc
            if cur_cost < best_cost:
                best = list(cur)
                best_cost = cur_cost
    return best, round(time.time()-t0, 2)

# ── AI Algorithm 2: Genetic Algorithm ────────────────────────────────


def genetic_algorithm(cells, nets, gw, gh, generations=80, pop_size=30):
    t0 = time.time()
    def make(): return random_placement(cells, gw, gh)
    def fit(ind): return 1.0/(cost(ind, nets, gw, gh)+1)

    def cross(p1, p2): cut = random.randint(
        1, len(p1)-1); return p1[:cut]+p2[cut:]

    def mutate(ind, rate=0.2):
        n = list(ind)
        for i in range(len(n)):
            if random.random() < rate:
                w, h = cells[i]['w'], cells[i]['h']
                n[i] = (random.randint(0, max(0, gw-w)),
                        random.randint(0, max(0, gh-h)), w, h)
        return n
    pop = [make() for _ in range(pop_size)]
    best = min(pop, key=lambda x: cost(x, nets, gw, gh))
    best_cost = cost(best, nets, gw, gh)
    for _ in range(generations):
        scored = sorted([(fit(ind), ind) for ind in pop], reverse=True)
        surv = [ind for _, ind in scored[:pop_size//2]]
        new_pop = list(surv)
        while len(new_pop) < pop_size:
            new_pop.append(
                mutate(cross(random.choice(surv), random.choice(surv))))
        pop = new_pop
        gb = min(pop, key=lambda x: cost(x, nets, gw, gh))
        gc = cost(gb, nets, gw, gh)
        if gc < best_cost:
            best = gb
            best_cost = gc
    return best, round(time.time()-t0, 2)

# ── AI Algorithm 3: Particle Swarm Optimization ───────────────────────


def particle_swarm(cells, nets, gw, gh, iterations=100, n_particles=20):
    t0 = time.time()
    n = len(cells)

    def to_pos(p):
        return [(int(max(0, min(gw-cells[i]['w'], p[i*2]))),
                 int(max(0, min(gh-cells[i]['h'], p[i*2+1]))),
                 cells[i]['w'], cells[i]['h']) for i in range(n)]

    def pcost(p): return cost(to_pos(p), nets, gw, gh)
    particles = [[float(random.randint(0, max(0, gw-cells[i//2]['w'] if i % 2 == 0 else gw-1)))
                  if i % 2 == 0 else float(random.randint(0, max(0, gh-cells[i//2]['h'])))
                  for i in range(n*2)] for _ in range(n_particles)]
    velocities = [[random.uniform(-2, 2) for _ in range(n*2)]
                  for _ in range(n_particles)]
    pb = [list(p) for p in particles]
    pbc = [pcost(p) for p in particles]
    gi = pbc.index(min(pbc))
    gb = list(pb[gi])
    gbc = pbc[gi]
    for _ in range(iterations):
        for i in range(n_particles):
            for d in range(n*2):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = max(-5, min(5, 0.7*velocities[i][d]+1.5*r1*(
                    pb[i][d]-particles[i][d])+1.5*r2*(gb[d]-particles[i][d])))
                particles[i][d] += velocities[i][d]
            pc = pcost(particles[i])
            if pc < pbc[i]:
                pb[i] = list(particles[i])
                pbc[i] = pc
            if pc < gbc:
                gb = list(particles[i])
                gbc = pc
    return to_pos(gb), round(time.time()-t0, 2)

# ── Renderer ──────────────────────────────────────────────────────────


def render_layout(cells, before_pos, after_pos, nets, circuit_name, gw, gh, algo_name, metrics):
    fig = plt.figure(figsize=(24, 14), facecolor='#030712')
    fig.text(0.5, 0.97, 'SmartEDA — AI-Driven VLSI Layout Generator',
             ha='center', fontsize=15, fontweight='bold', color='#00d4ff', fontfamily='monospace')
    fig.text(0.5, 0.935, f'Circuit: {circuit_name}   |   Cells: {len(cells)}   |   Nets: {len(nets)}   |   Algorithm: {algo_name}',
             ha='center', fontsize=9, color='#64748b', fontfamily='monospace')
    ax1 = fig.add_axes([0.03, 0.20, 0.44, 0.70])
    ax2 = fig.add_axes([0.53, 0.20, 0.44, 0.70])

    def draw(ax, pos, title, highlight_ov):
        ax.set_facecolor('#060d18')
        ax.set_xlim(-0.5, gw+0.5)
        ax.set_ylim(-0.5, gh+0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(gw+1))
        ax.set_yticks(range(gh+1))
        ax.tick_params(colors='#1e3a5f', labelsize=5)
        ax.grid(True, color='#1e3a5f', linewidth=0.3, alpha=0.4)
        for sp in ax.spines.values():
            sp.set_edgecolor('#1e3a5f')
        ax.add_patch(mpatches.FancyBboxPatch((0, 0), gw, gh, boxstyle="square,pad=0",
                     lw=2, edgecolor='#00d4ff', facecolor='none', linestyle='--', alpha=0.6))
        for i, j in nets:
            if i >= len(pos) or j >= len(pos):
                continue
            x1, y1, w1, h1 = pos[i]
            x2, y2, w2, h2 = pos[j]
            ax.plot([x1+w1/2, x2+w2/2], [y1+h1/2, y2+h2/2],
                    color='#ff4444' if highlight_ov else '#00ff88', lw=0.6, alpha=0.3, zorder=1)
        for idx, c in enumerate(cells):
            if idx >= len(pos):
                continue
            x, y, w, h = pos[idx]
            col = CELL_COLORS.get(c['type'], '#555')
            has_ov = highlight_ov and any(idx != j and pos[j][0] < x+w and pos[j][0]+pos[j][2]
                                          > x and pos[j][1] < y+h and pos[j][1]+pos[j][3] > y for j in range(len(pos)))
            ax.add_patch(mpatches.FancyBboxPatch((x+0.06, y+0.06), w-0.12, h-0.12, boxstyle="round,pad=0.05",
                         lw=1.5 if has_ov else 1.2, edgecolor='#ff4444' if has_ov else col, facecolor=col+'44', zorder=2))
            ax.text(x+w/2, y+h/2, c['name'], ha='center', va='center', fontsize=min(8, max(5, w*1.8)), fontweight='bold',
                    color='white', zorder=3, path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
            if h > 1.4:
                ax.text(x+w/2, y+0.22, c['type'], ha='center',
                        fontsize=4.5, color=col, alpha=0.75, zorder=3)
        wl = wirelength(pos, nets)
        ov = count_overlaps(pos)
        drc_col = '#00ff88' if ov == 0 else '#ff4444'
        ax.set_title(title, color='white', fontsize=9,
                     fontweight='bold', pad=6, fontfamily='monospace')
        ax.text(0.02, 0.02, f'WL: {wl:.0f}  |  DRC: {"✓ PASS" if ov == 0 else f"✗ {ov} violations"}', transform=ax.transAxes,
                fontsize=7, color=drc_col, fontfamily='monospace', bbox=dict(facecolor='#000000aa', edgecolor='none', pad=3))

    wl_b = wirelength(before_pos, nets)
    wl_a = wirelength(after_pos, nets)
    ov_b = count_overlaps(before_pos)
    ov_a = count_overlaps(after_pos)
    draw(ax1, before_pos, f'BEFORE — Random Placement  (WL: {wl_b:.0f})', True)
    draw(ax2, after_pos, f'AFTER  — {algo_name}  (WL: {wl_a:.0f})', False)

    mx = fig.add_axes([0.03, 0.01, 0.94, 0.16])
    mx.set_facecolor('#0d1117')
    mx.set_xlim(0, 1)
    mx.set_ylim(0, 1)
    mx.axis('off')
    for sp in mx.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('#1e3a5f')
    wl_imp = (wl_b-wl_a)/wl_b*100 if wl_b > 0 else 0
    pwr_b = estimate_power(wl_b, cells)
    pwr_a = estimate_power(wl_a, cells)
    pwr_imp = (pwr_b-pwr_a)/pwr_b*100 if pwr_b > 0 else 0
    tim_b = estimate_timing(wl_b, nets)
    tim_a = estimate_timing(wl_a, nets)
    tim_imp = (tim_b-tim_a)/max(tim_b, 0.1)*100
    au = area_utilization(after_pos, gw, gh)
    cards = [
        ("Wire Length", f"{wl_b:.0f} u",
         f"{wl_a:.0f} u", f"▼ {wl_imp:.1f}%", "#00d4ff"),
        ("DRC Violations", f"{ov_b}", f"{ov_a}",
         f"▼ {(ov_b-ov_a)/max(ov_b, 1)*100:.1f}%", "#00ff88"),
        ("Power (Est.)", f"{pwr_b:.0f} mW",
         f"{pwr_a:.0f} mW", f"▼ {pwr_imp:.1f}%", "#7c3aed"),
        ("Timing (Est.)", f"{tim_b:.1f} ns",
         f"{tim_a:.1f} ns", f"▼ {tim_imp:.1f}%", "#ffd700"),
        ("Area Util.", "Unopt.", f"{au}%", "Optimized", "#f97316"),
        ("Opt. Time", "Manual: hrs",
         f"AI: {metrics['elapsed']}s", "✓ Auto", "#06b6d4"),
    ]
    cw = 1/len(cards)
    for i, (lbl, bef, aft, imp, col) in enumerate(cards):
        cx = i*cw+cw/2
        mx.add_patch(mpatches.FancyBboxPatch((i*cw+0.006, 0.06), cw-0.012, 0.88,
                     boxstyle="round,pad=0.01", facecolor=col+'11', edgecolor=col+'44', lw=1))
        mx.text(cx, .87, lbl, ha='center', fontsize=8,
                color='#94a3b8', fontfamily='monospace')
        mx.text(cx, .65, bef, ha='center', fontsize=9, color='#ff6b6b',
                fontfamily='monospace', fontweight='bold')
        mx.text(cx, .44, '↓', ha='center', fontsize=11, color=col)
        mx.text(cx, .24, aft, ha='center', fontsize=9, color=col,
                fontfamily='monospace', fontweight='bold')
        mx.text(cx, .07, imp, ha='center', fontsize=8, color='#00ff88',
                fontfamily='monospace', fontweight='bold')
    leg_x = 0.03
    for ctype, col in CELL_COLORS.items():
        fig.text(leg_x, 0.002, f'■ {ctype}', fontsize=6.5,
                 color=col, fontfamily='monospace')
        leg_x += 0.115
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130,
                bbox_inches='tight', facecolor='#030712')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ── Routes ────────────────────────────────────────────────────────────


@app.route('/')
def index():
    return render_template('index.html', cell_types=CELL_TYPES, presets=PRESETS)


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        cells = data.get('cells', [])
        connections = data.get('connections', [])
        circuit_name = data.get('circuit_name', 'Custom Circuit')
        grid_w = int(data.get('grid_w', 24))
        grid_h = int(data.get('grid_h', 20))
        algorithm = data.get('algorithm', 'sa')
        if len(cells) < 2:
            return jsonify({'error': 'Please add at least 2 cells.'}), 400
        if len(connections) < 1:
            return jsonify({'error': 'Please add at least 1 connection.'}), 400
        names = {c['name'] for c in cells}
        for a, b in connections:
            if a not in names:
                return jsonify({'error': f'Cell "{a}" not found.'}), 400
            if b not in names:
                return jsonify({'error': f'Cell "{b}" not found.'}), 400
        nets = build_nets(cells, connections)
        before_pos = random_placement(cells, grid_w, grid_h)
        algo_names = {'sa': 'Simulated Annealing',
                      'ga': 'Genetic Algorithm', 'pso': 'Particle Swarm Optimization'}
        algo_name = algo_names.get(algorithm, 'Simulated Annealing')
        if algorithm == 'ga':
            after_pos, elapsed = genetic_algorithm(cells, nets, grid_w, grid_h)
        elif algorithm == 'pso':
            after_pos, elapsed = particle_swarm(cells, nets, grid_w, grid_h)
        else:
            after_pos, elapsed = simulated_annealing(
                cells, nets, grid_w, grid_h)
        wl_b = wirelength(before_pos, nets)
        wl_a = wirelength(after_pos, nets)
        ov_b = count_overlaps(before_pos)
        ov_a = count_overlaps(after_pos)
        wl_imp = round((wl_b-wl_a)/wl_b*100, 1) if wl_b > 0 else 0
        pwr_b = estimate_power(wl_b, cells)
        pwr_a = estimate_power(wl_a, cells)
        pwr_imp = round((pwr_b-pwr_a)/pwr_b*100, 1) if pwr_b > 0 else 0
        tim_b = estimate_timing(wl_b, nets)
        tim_a = estimate_timing(wl_a, nets)
        au = area_utilization(after_pos, grid_w, grid_h)
        metrics = {'wl_before': round(wl_b, 1), 'wl_after': round(wl_a, 1), 'wl_improve': wl_imp,
                   'ov_before': ov_b, 'ov_after': ov_a, 'drc': 'PASS' if ov_a == 0 else f'FAIL ({ov_a})',
                   'power_before': pwr_b, 'power_after': pwr_a, 'power_improve': pwr_imp,
                   'timing_before': tim_b, 'timing_after': tim_a, 'area_util': au,
                   'elapsed': elapsed, 'cells': len(cells), 'nets': len(nets), 'algorithm': algo_name}
        app.config['last_positions'] = after_pos
        app.config['last_cells'] = cells
        app.config['last_metrics'] = metrics
        app.config['last_circuit'] = circuit_name
        img_b64 = render_layout(cells, before_pos, after_pos,
                                nets, circuit_name, grid_w, grid_h, algo_name, metrics)
        return jsonify({'image': img_b64, 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare():
    try:
        data = request.get_json()
        cells = data.get('cells', [])
        connections = data.get('connections', [])
        grid_w = int(data.get('grid_w', 24))
        grid_h = int(data.get('grid_h', 20))
        if len(cells) < 2 or len(connections) < 1:
            return jsonify({'error': 'Need at least 2 cells and 1 connection.'}), 400
        nets = build_nets(cells, connections)
        results = {}
        for algo, name in [('sa', 'Simulated Annealing'), ('ga', 'Genetic Algorithm'), ('pso', 'Particle Swarm')]:
            if algo == 'ga':
                pos, elapsed = genetic_algorithm(cells, nets, grid_w, grid_h)
            elif algo == 'pso':
                pos, elapsed = particle_swarm(cells, nets, grid_w, grid_h)
            else:
                pos, elapsed = simulated_annealing(cells, nets, grid_w, grid_h)
            wl = wirelength(pos, nets)
            ov = count_overlaps(pos)
            results[algo] = {'name': name, 'wl': round(wl, 1), 'overlaps': ov,
                             'drc': 'PASS' if ov == 0 else f'FAIL ({ov})',
                             'power': estimate_power(wl, cells), 'timing': estimate_timing(wl, nets), 'elapsed': elapsed}
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export/csv')
def export_csv():
    try:
        positions = app.config.get('last_positions', [])
        cells = app.config.get('last_cells', [])
        metrics = app.config.get('last_metrics', {})
        circuit = app.config.get('last_circuit', 'circuit')
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['SmartEDA - VLSI Layout Export'])
        writer.writerow(['Circuit', circuit])
        writer.writerow(['Algorithm', metrics.get('algorithm', 'N/A')])
        writer.writerow([])
        writer.writerow(['=== CELL PLACEMENTS ==='])
        writer.writerow(['Cell Name', 'Type', 'X', 'Y', 'Width', 'Height'])
        for i, c in enumerate(cells):
            if i < len(positions):
                x, y, w, h = positions[i]
                writer.writerow([c['name'], c['type'], x, y, w, h])
        writer.writerow([])
        writer.writerow(['=== PPA METRICS ==='])
        writer.writerow(['Metric', 'Before', 'After', 'Improvement'])
        writer.writerow(['Wire Length', metrics.get('wl_before'), metrics.get(
            'wl_after'), f"{metrics.get('wl_improve')}%"])
        writer.writerow(['DRC Violations', metrics.get(
            'ov_before'), metrics.get('ov_after'), ''])
        writer.writerow(['Power (mW)', metrics.get('power_before'), metrics.get(
            'power_after'), f"{metrics.get('power_improve')}%"])
        writer.writerow(['Timing (ns)', metrics.get(
            'timing_before'), metrics.get('timing_after'), ''])
        writer.writerow(
            ['Area Utilization', '', f"{metrics.get('area_util')}%", ''])
        writer.writerow(['Optimization Time (s)', '',
                        metrics.get('elapsed'), ''])
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv',
                         as_attachment=True, download_name=f'smarteda_{circuit.replace(" ", "_")}.csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/preset/<pname>')
def preset(pname):
    if pname not in PRESETS:
        return jsonify({'error': 'Preset not found'}), 404
    return jsonify(PRESETS[pname])


if __name__ == '__main__':
    print("\n  SmartEDA Extended — Running!")
    print("  Open browser:  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
