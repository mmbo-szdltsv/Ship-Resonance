import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
from matplotlib.figure import Figure
import matplotlib.patches as patches


class MembershipFunctions:
    @staticmethod
    def linear(value, min_val, max_val):
        if value <= min_val:
            return 0.0
        if value >= max_val:
            return 1.0
        return (value - min_val) / (max_val - min_val)

    @staticmethod
    def triangular(value, left, center, right):
        if value <= left or value >= right:
            return 0.0
        if value == center:
            return 1.0
        if value < center:
            return (value - left) / (center - left)
        return (right - value) / (right - center)

    @staticmethod
    def calculate_course_correction(alpha, min_angle, max_angle):
        if alpha <= 0:
            return 0.0
        numerator = 0.0
        denominator = 0.0
        step = 0.1
        for angle in np.arange(min_angle, max_angle + step, step):
            mu = (angle - min_angle) / (max_angle - min_angle)
            mu_final = min(mu, alpha)
            numerator += angle * mu_final
            denominator += mu_final
        return numerator / denominator if denominator > 0 else 0.0


class ProbabilityModels:
    @staticmethod
    def bayes_model(prob_e, prob_he, prob_h_not_e):
        prob_not_e = 1 - prob_e
        return prob_he * prob_e + prob_h_not_e * prob_not_e

    @staticmethod
    def shortliffe_model(confidence_he1, confidence_he2):
        return confidence_he1 + confidence_he2 * (1 - confidence_he1)


class RuleSystem:
    class Rule:
        def __init__(self, rule_type, name, description, min_freq_ratio, max_freq_ratio, min_amplitude, confidence):
            self.type = rule_type
            self.name = name
            self.description = description
            self.min_freq_ratio = min_freq_ratio
            self.max_freq_ratio = max_freq_ratio
            self.min_amplitude = min_amplitude
            self.confidence = confidence

    class RuleEvaluationResult:
        def __init__(self, rule, is_activated, activation_degree, final_confidence):
            self.rule = rule
            self.is_activated = is_activated
            self.activation_degree = activation_degree
            self.final_confidence = final_confidence

    def __init__(self):
        self.rules = []
        self.initialize_rules()

    def initialize_rules(self):
        self.rules.append(self.Rule(
            rule_type="main_roll",
            name="Основной бортовой резонанс",
            description="Совпадение частоты волнения и собственной частоты бортовой качки",
            min_freq_ratio=0.8,
            max_freq_ratio=1.2,
            min_amplitude=15.0,
            confidence=0.95
        ))
        self.rules.append(self.Rule(
            rule_type="parametric_roll",
            name="Параметрический резонанс",
            description="Частота волнения ≈ удвоенной собственной частоте бортовой качки",
            min_freq_ratio=1.9,
            max_freq_ratio=2.0,
            min_amplitude=15.0,
            confidence=0.85
        ))
        self.rules.append(self.Rule(
            rule_type="main_pitch",
            name="Основной килевой резонанс",
            description="Совпадение частоты волнения и собственной частоты килевой качки",
            min_freq_ratio=0.8,
            max_freq_ratio=1.2,
            min_amplitude=3.0,
            confidence=0.90
        ))

    def evaluate_rules(self, roll_amplitude, pitch_amplitude, roll_ratio, pitch_ratio):
        results = []
        for rule in self.rules:
            is_activated = False
            activation_degree = 0.0
            if rule.type == "main_roll":
                if roll_amplitude >= rule.min_amplitude and rule.min_freq_ratio <= roll_ratio <= rule.max_freq_ratio:
                    is_activated = True
                    activation_degree = self.calculate_activation_degree(
                        roll_amplitude, rule.min_amplitude, roll_ratio, rule.min_freq_ratio, rule.max_freq_ratio
                    )
            elif rule.type == "parametric_roll":
                if roll_amplitude >= rule.min_amplitude and rule.min_freq_ratio <= roll_ratio <= rule.max_freq_ratio:
                    is_activated = True
                    activation_degree = self.calculate_activation_degree(
                        roll_amplitude, rule.min_amplitude, roll_ratio, rule.min_freq_ratio, rule.max_freq_ratio
                    )
            elif rule.type == "main_pitch":
                if pitch_amplitude >= rule.min_amplitude and rule.min_freq_ratio <= pitch_ratio <= rule.max_freq_ratio:
                    is_activated = True
                    activation_degree = self.calculate_activation_degree(
                        pitch_amplitude, rule.min_amplitude, pitch_ratio, rule.min_freq_ratio, rule.max_freq_ratio
                    )
            final_confidence = rule.confidence * activation_degree if is_activated else 0.0
            results.append(self.RuleEvaluationResult(rule, is_activated, activation_degree, final_confidence))
        return results

    def calculate_activation_degree(self, amplitude, min_amplitude, freq_ratio, min_freq, max_freq):
        amplitude_degree = min(1.0, amplitude / min_amplitude)
        center_freq = (min_freq + max_freq) / 2
        freq_degree = 1.0 - abs(freq_ratio - center_freq) / ((max_freq - min_freq) / 2)
        return min(amplitude_degree, freq_degree)


class ShipRollingControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Контроль резонансных режимов качки (студент)")
        self.root.geometry("1100x750")

        # === Ваши параметры ===
        self.variant_params = {
            'ship_length': 45.0,
            'ship_width': 8.2,
            'ship_draft': 3.5,
            'metacenter_height': 0.52,
            'ship_speed': 11.5,
            'theta_m': 15.0,
            'psi_m': 3.0,
            'prob_e': 0.75,
            'prob_he': 0.9,
            'prob_h_not_e': 0.01,
            'confidence_he1': 0.89,
            'confidence_he2': 0.97,
            'roll_amp_range': (12, 20),
            'pitch_amp_range': (2.5, 4.5),
            'main_roll_freq_range': (0.8, 1.2),
            'main_pitch_freq_range': (0.8, 1.2),
            'parametric_roll_freq_range': (1.9, 2.0),
            'course_angle_range': (0, 30)
        }

        self.natural_roll_period = 0.0
        self.natural_pitch_period = 0.0
        self.apparent_wave_period = 0.0
        self.wave_speed = 0.0
        self.wave_length = 0.0

        self.create_widgets()
        self.load_default_values()

    def create_widgets(self):
        # Главная вкладка
        self.tab_control = ttk.Notebook(self.root)
        self.tab_main = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_main, text='Главная')
        self.tab_diagram = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_diagram, text='Диаграмма')
        self.tab_membership = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_membership, text='Функции принадлежности')
        self.tab_control.pack(expand=1, fill='both', padx=5, pady=5)

        # Левая панель — параметры
        left_frame = ttk.Frame(self.tab_main)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Параметры корабля
        ship_frame = ttk.LabelFrame(left_frame, text="Корабль", padding=5)
        ship_frame.pack(fill=tk.X, pady=5)
        labels_ship = ["Длина (м):", "Ширина (м):", "Осадка (м):", "Метац. высота (м):", "Скорость (уз):"]
        self.ship_vars = []
        for i, label in enumerate(labels_ship):
            row = ttk.Frame(ship_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            var = tk.DoubleVar()
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)
            self.ship_vars.append(var)

        # Параметры качки
        wave_frame = ttk.LabelFrame(left_frame, text="Качка", padding=5)
        wave_frame.pack(fill=tk.X, pady=5)
        labels_wave = ["Длина волны (м):", "θ (ампл., °):", "ψ (ампл., °):", "Курсовой угол (°):"]
        self.wave_vars = []
        for i, label in enumerate(labels_wave):
            row = ttk.Frame(wave_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            var = tk.DoubleVar()
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)
            self.wave_vars.append(var)

        # Вероятности
        prob_frame = ttk.LabelFrame(left_frame, text="Вероятности", padding=5)
        prob_frame.pack(fill=tk.X, pady=5)
        labels_prob = ["P(E):", "P(H/E):", "P(H/¬E):", "МД(H/E1):", "МД(H/E2):"]
        self.prob_vars = []
        for i, label in enumerate(labels_prob):
            row = ttk.Frame(prob_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            var = tk.DoubleVar()
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)
            self.prob_vars.append(var)

        # Кнопки
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Анализировать", command=self.analyze).pack(pady=2)
        ttk.Button(btn_frame, text="Показать вероятности", command=self.show_probabilities).pack(pady=2)
        ttk.Button(btn_frame, text="Сбросить", command=self.load_default_values).pack(pady=2)

        # Результаты (справа)
        results_frame = ttk.LabelFrame(self.tab_main, text="Результаты", padding=5)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, font=('Courier New', 9), height=30)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Диаграмма
        self.fig_diagram = Figure(figsize=(9, 7), dpi=100)
        self.ax_diagram = self.fig_diagram.add_subplot(111)
        self.canvas_diagram = FigureCanvasTkAgg(self.fig_diagram, master=self.tab_diagram)
        self.canvas_diagram.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Функции принадлежности
        self.fig_membership = Figure(figsize=(9, 7), dpi=100)
        self.canvas_membership = FigureCanvasTkAgg(self.fig_membership, master=self.tab_membership)
        self.canvas_membership.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_default_values(self):
        self.ship_vars[0].set(self.variant_params['ship_length'])
        self.ship_vars[1].set(self.variant_params['ship_width'])
        self.ship_vars[2].set(self.variant_params['ship_draft'])
        self.ship_vars[3].set(self.variant_params['metacenter_height'])
        self.ship_vars[4].set(self.variant_params['ship_speed'])

        self.wave_vars[0].set(self.variant_params['ship_length'])  # λ = L
        self.wave_vars[1].set(16.0)
        self.wave_vars[2].set(3.2)
        self.wave_vars[3].set(20.0)

        self.prob_vars[0].set(self.variant_params['prob_e'])
        self.prob_vars[1].set(self.variant_params['prob_he'])
        self.prob_vars[2].set(self.variant_params['prob_h_not_e'])
        self.prob_vars[3].set(self.variant_params['confidence_he1'])
        self.prob_vars[4].set(self.variant_params['confidence_he2'])

        self.results_text.delete(1.0, tk.END)
        self.plot_diagram()
        self.plot_membership_functions()

    def get_input_data(self):
        return {
            'ship_length': self.ship_vars[0].get(),
            'ship_width': self.ship_vars[1].get(),
            'ship_draft': self.ship_vars[2].get(),
            'metacenter_height': self.ship_vars[3].get(),
            'ship_speed': self.ship_vars[4].get(),
            'wave_length': self.wave_vars[0].get(),
            'roll_amplitude': self.wave_vars[1].get(),
            'pitch_amplitude': self.wave_vars[2].get(),
            'course_angle': self.wave_vars[3].get(),
            'prob_e': self.prob_vars[0].get(),
            'prob_he': self.prob_vars[1].get(),
            'prob_h_not_e': self.prob_vars[2].get(),
            'confidence_he1': self.prob_vars[3].get(),
            'confidence_he2': self.prob_vars[4].get()
        }

    def perform_calculations(self, data):
        inertia_coeff = 0.8
        self.natural_roll_period = (inertia_coeff * data['ship_width']) / math.sqrt(data['metacenter_height'])
        self.natural_pitch_period = 2.5 * math.sqrt(data['ship_draft'])
        self.wave_length = data['wave_length']
        self.wave_speed = 1.25 * math.sqrt(self.wave_length)

        ship_speed_ms = data['ship_speed'] * 0.51444
        course_rad = math.radians(data['course_angle'])
        v_cos_phi = ship_speed_ms * math.cos(course_rad)

        if abs(self.wave_speed - v_cos_phi) < 0.1:
            self.apparent_wave_period = float('inf')
        else:
            self.apparent_wave_period = self.wave_length / (self.wave_speed - v_cos_phi)

        roll_ratio = self.natural_roll_period / self.apparent_wave_period if self.apparent_wave_period > 0 else 0
        pitch_ratio = self.natural_pitch_period / self.apparent_wave_period if self.apparent_wave_period > 0 else 0
        return roll_ratio, pitch_ratio

    def analyze(self):
        data = self.get_input_data()
        roll_ratio, pitch_ratio = self.perform_calculations(data)

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Собств. период θ: {self.natural_roll_period:.2f} с\n")
        self.results_text.insert(tk.END, f"Собств. период ψ: {self.natural_pitch_period:.2f} с\n")
        self.results_text.insert(tk.END, f"Скорость волны: {self.wave_speed:.2f} м/с\n")
        self.results_text.insert(tk.END, f"Кажущийся период: {self.apparent_wave_period:.2f} с\n")
        self.results_text.insert(tk.END, f"σ/ωθ = {roll_ratio:.3f}\n")
        self.results_text.insert(tk.END, f"σ/ωψ = {pitch_ratio:.3f}\n")
        self.results_text.insert(tk.END, "-" * 40 + "\n")

        rule_system = RuleSystem()
        rule_results = rule_system.evaluate_rules(data['roll_amplitude'], data['pitch_amplitude'], roll_ratio, pitch_ratio)

        bayes_prob = ProbabilityModels.bayes_model(data['prob_e'], data['prob_he'], data['prob_h_not_e'])
        shortliffe_conf = ProbabilityModels.shortliffe_model(data['confidence_he1'], data['confidence_he2'])

        self.results_text.insert(tk.END, f"Вероятность (Байес): {bayes_prob:.4f}\n")
        self.results_text.insert(tk.END, f"Уверенность (Шортлифф): {shortliffe_conf:.4f}\n\n")

        resonance_detected = False
        for result in rule_results:
            if result.is_activated:
                resonance_detected = True
                self.results_text.insert(tk.END, f"!!! {result.rule.name.upper()} !!!\n")
                self.results_text.insert(tk.END, f"{result.rule.description}\n")
                self.results_text.insert(tk.END, f"Активация: {result.activation_degree:.3f}, Уверенность: {result.final_confidence:.3f}\n\n")

        if not resonance_detected:
            self.results_text.insert(tk.END, "РЕЗОНАНС НЕ ОБНАРУЖЕН\n\n")

        self.perform_fuzzy_analysis(data, roll_ratio, pitch_ratio)
        self.plot_diagram(data)

    def perform_fuzzy_analysis(self, data, roll_ratio, pitch_ratio):
        mu_roll = MembershipFunctions.linear(data['roll_amplitude'], *self.variant_params['roll_amp_range'])
        mu_pitch = MembershipFunctions.linear(data['pitch_amplitude'], *self.variant_params['pitch_amp_range'])
        mu_main_roll = MembershipFunctions.triangular(roll_ratio, 0.8, 1.0, 1.2)
        mu_param = MembershipFunctions.triangular(
            roll_ratio,
            self.variant_params['parametric_roll_freq_range'][0],
            sum(self.variant_params['parametric_roll_freq_range']) / 2,
            self.variant_params['parametric_roll_freq_range'][1]
        )
        mu_main_pitch = MembershipFunctions.triangular(pitch_ratio, 0.8, 1.0, 1.2)

        alpha1 = min(mu_roll, mu_main_roll)
        alpha2 = min(mu_roll, mu_param)
        alpha3 = min(mu_pitch, mu_main_pitch)

        self.results_text.insert(tk.END, "Нечёткий вывод:\n")
        if alpha1 > 0:
            self.results_text.insert(tk.END, f"Правило 1 активно (α={alpha1:.3f})\n")
        if alpha2 > 0:
            self.results_text.insert(tk.END, f"Правило 2 активно (α={alpha2:.3f})\n")
        if alpha3 > 0:
            self.results_text.insert(tk.END, f"Правило 3 активно (α={alpha3:.3f})\n")

        if any(a > 0 for a in [alpha1, alpha2, alpha3]):
            max_alpha = max(alpha1, alpha2, alpha3)
            new_course = MembershipFunctions.calculate_course_correction(
                max_alpha,
                *self.variant_params['course_angle_range']
            )
            self.results_text.insert(tk.END, f"\nРЕКОМЕНДАЦИЯ: изменить курс на {new_course:.0f}°\n")
        else:
            self.results_text.insert(tk.END, "\nДефаззификация не требуется.\n")

    def show_probabilities(self):
        data = self.get_input_data()
        bayes = ProbabilityModels.bayes_model(data['prob_e'], data['prob_he'], data['prob_h_not_e'])
        short = ProbabilityModels.shortliffe_model(data['confidence_he1'], data['confidence_he2'])
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Расчёт вероятностей:\n")
        self.results_text.insert(tk.END, f"P(H) = P(H|E)*P(E) + P(H|¬E)*P(¬E)\n")
        self.results_text.insert(tk.END, f"P(H) = {data['prob_he']}*{data['prob_e']} + {data['prob_h_not_e']}*{1-data['prob_e']} = {bayes:.4f}\n\n")
        self.results_text.insert(tk.END, "МД(H) = МД(H|E1) + МД(H|E2)*(1 - МД(H|E1))\n")
        self.results_text.insert(tk.END, f"МД(H) = {data['confidence_he1']} + {data['confidence_he2']}*(1 - {data['confidence_he1']}) = {short:.4f}\n")

    def plot_diagram(self, data=None):
        if data is None:
            data = self.get_input_data()
        self.ax_diagram.clear()
        R = 150
        self.ax_diagram.set_xlim(-R * 1.2, R * 1.2)
        self.ax_diagram.set_ylim(-10, R * 1.2)
        self.ax_diagram.set_aspect('equal')
        self.ax_diagram.axis('off')

        # Основной круг
        circle = plt.Circle((0, 0), R, fill=False, color='black', linewidth=1.2)
        self.ax_diagram.add_patch(circle)

        # Концентрические окружности (скорость)
        for v in range(5, 26, 5):
            r = R * v / 25
            c = plt.Circle((0, 0), r, fill=False, color='#888888', linestyle='--', linewidth=0.6)
            self.ax_diagram.add_patch(c)
            self.ax_diagram.text(r * 0.7, r * 0.7, f'{v}', fontsize=7, color='#444444')

        # Лучи (курсовые углы)
        for deg in range(0, 181, 30):
            rad = math.radians(deg)
            x, y = R * math.cos(rad), R * math.sin(rad)
            self.ax_diagram.plot([0, x], [0, y], color='#666666', linestyle='--', linewidth=0.6)
            self.ax_diagram.text(x * 1.05, y * 1.05, f'{deg}°', fontsize=8, color='#444444')

        # Резонансные зоны с новыми цветами
        if self.wave_length > 0 and self.natural_roll_period > 0:
            self.draw_resonance_zone(0.8, 1.2, self.natural_roll_period, '#555555', 0.25, 'Осн. борт.')
            self.draw_resonance_zone(1.9, 2.0, self.natural_roll_period, '#8B4513', 0.25, 'Пар. борт.')
            self.draw_resonance_zone(0.8, 1.2, self.natural_pitch_period, '#2E8B57', 0.25, 'Осн. кил.')

        # Положение корабля
        v = data['ship_speed']
        phi = data['course_angle']
        if 0 < v <= 25:
            r = R * v / 25
            rad = math.radians(phi)
            x, y = r * math.cos(rad), r * math.sin(rad)
            self.ax_diagram.plot([0, x], [0, y], color='#006400', linewidth=2)  # тёмно-зелёный
            self.ax_diagram.plot(x, y, 'o', color='#006400', markersize=6)
            self.ax_diagram.text(x * 1.1, y * 1.1, f"V={v:.1f}\nφ={phi:.0f}°", fontsize=8, color='#006400')

        self.ax_diagram.set_title("Диаграмма качки", fontsize=10)
        self.fig_diagram.tight_layout()
        self.canvas_diagram.draw_idle()

    def draw_resonance_zone(self, k_min, k_max, period, color, alpha, label):
        if self.wave_speed == 0 or self.wave_length == 0:
            return
        try:
            Cw_knots = self.wave_speed / 0.5144
            tau1 = period / k_min
            v1 = Cw_knots - (self.wave_length / 0.5144) / tau1
            tau2 = period / k_max
            v2 = Cw_knots - (self.wave_length / 0.5144) / tau2
            R = 150
            scale = R / 25
            x1, x2 = v1 * scale, v2 * scale
            x_left, x_right = min(x1, x2), max(x1, x2)
            if x_right - x_left > 0:
                theta1 = math.acos(max(-1, min(1, x_left / R)))
                theta2 = math.acos(max(-1, min(1, x_right / R)))
                theta = np.linspace(theta1, theta2, 30)
                arc_x = R * np.cos(theta)
                arc_y = R * np.sin(theta)
                xs = np.concatenate([[x_left, x_right], arc_x[::-1]])
                ys = np.concatenate([[0, 0], arc_y[::-1]])
                self.ax_diagram.fill(xs, ys, color=color, alpha=alpha, edgecolor=color, linewidth=0.8)

                # Подпись с рамкой в цвете зоны
                center_x = (x_left + x_right) / 2
                center_y = R * 0.6
                self.ax_diagram.text(
                    center_x, center_y, label,
                    ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.2'),
                    color='black'
                )
        except:
            pass

    def plot_membership_functions(self):
        self.fig_membership.clear()
        gs = self.fig_membership.add_gridspec(2, 3)

        # θ
        ax = self.fig_membership.add_subplot(gs[0, 0])
        x = np.linspace(0, 30, 300)
        y = [MembershipFunctions.linear(v, 12, 20) for v in x]
        ax.plot(x, y, 'b-')
        ax.set_title('θ (амплитуда)')
        ax.grid(True, alpha=0.3)

        # ψ
        ax = self.fig_membership.add_subplot(gs[0, 1])
        x = np.linspace(0, 6, 300)
        y = [MembershipFunctions.linear(v, 2.5, 4.5) for v in x]
        ax.plot(x, y, 'g-')
        ax.set_title('ψ (амплитуда)')
        ax.grid(True, alpha=0.3)

        # σ/ωθ основной
        ax = self.fig_membership.add_subplot(gs[0, 2])
        x = np.linspace(0.5, 1.5, 300)
        y = [MembershipFunctions.triangular(v, 0.8, 1.0, 1.2) for v in x]
        ax.plot(x, y, 'r-')
        ax.set_title('σ/ωθ (осн.)')
        ax.grid(True, alpha=0.3)

        # σ/ωθ параметрический
        ax = self.fig_membership.add_subplot(gs[1, 0])
        x = np.linspace(1.5, 2.5, 300)
        y = [MembershipFunctions.triangular(v, 1.9, 1.95, 2.0) for v in x]
        ax.plot(x, y, 'orange')
        ax.set_title('σ/ωθ (парам.)')
        ax.grid(True, alpha=0.3)

        # σ/ωψ
        ax = self.fig_membership.add_subplot(gs[1, 1])
        x = np.linspace(0.5, 1.5, 300)
        y = [MembershipFunctions.triangular(v, 0.8, 1.0, 1.2) for v in x]
        ax.plot(x, y, 'purple')
        ax.set_title('σ/ωψ')
        ax.grid(True, alpha=0.3)

        self.fig_membership.delaxes(self.fig_membership.add_subplot(gs[1, 2]))
        self.fig_membership.tight_layout()
        self.canvas_membership.draw_idle()


def main():
    root = tk.Tk()
    app = ShipRollingControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()