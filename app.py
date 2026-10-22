import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pickle
import pandas as pd
from datetime import datetime

class HotelBookingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Аналітична система прогнозування скасування бронювання")
        self.root.geometry("620x650")
        
        # Завантаження збережених моделей
        self.models = {}
        model_files = {
            "Logistic Regression": "model_logistic_regression.pkl",
            "Decision Tree": "model_decision_tree.pkl",
            "K-NN": "model_knn.pkl"
        }
        
        for name, file_path in model_files.items():
            try:
                with open(file_path, "rb") as f:
                    self.models[name] = pickle.load(f)
            except Exception as e:
                print(f"Попередження: Не вдалося завантажити {file_path}: {e}")

        # Елементи форматування
        tk.Label(root, text="Параметри готельного бронювання", font=("Arial", 14, "bold")).pack(pady=10)
        
        form_frame = tk.Frame(root)
        form_frame.pack(pady=5, padx=10, fill="x")

        # Поля введення
        tk.Label(form_frame, text="Днів до заїзду (Lead Time):").grid(row=0, column=0, sticky="w", pady=4)
        self.entry_lead_time = tk.Entry(form_frame)
        self.entry_lead_time.grid(row=0, column=1, pady=4)
        self.entry_lead_time.insert(0, "45")

        tk.Label(form_frame, text="Середня ціна за ніч (ADR):").grid(row=1, column=0, sticky="w", pady=4)
        self.entry_adr = tk.Entry(form_frame)
        self.entry_adr.grid(row=1, column=1, pady=4)
        self.entry_adr.insert(0, "120.0")

        tk.Label(form_frame, text="Кількість дорослих (Adults):").grid(row=2, column=0, sticky="w", pady=4)
        self.entry_adults = tk.Entry(form_frame)
        self.entry_adults.grid(row=2, column=1, pady=4)
        self.entry_adults.insert(0, "2")

        tk.Label(form_frame, text="Попередні скасування:").grid(row=3, column=0, sticky="w", pady=4)
        self.entry_prev_cancellations = tk.Entry(form_frame)
        self.entry_prev_cancellations.grid(row=3, column=1, pady=4)
        self.entry_prev_cancellations.insert(0, "0")

        tk.Label(form_frame, text="Тип готелю:").grid(row=4, column=0, sticky="w", pady=4)
        self.combo_hotel = ttk.Combobox(form_frame, values=["City Hotel", "Resort Hotel"], state="readonly")
        self.combo_hotel.grid(row=4, column=1, pady=4)
        self.combo_hotel.current(0)

        tk.Label(form_frame, text="Тип депозиту:").grid(row=5, column=0, sticky="w", pady=4)
        self.combo_deposit = ttk.Combobox(form_frame, values=["No Deposit", "Non Refund", "Refundable"], state="readonly")
        self.combo_deposit.grid(row=5, column=1, pady=4)
        self.combo_deposit.current(0)

        # Кнопка запуску прогнозування
        btn_predict = tk.Button(root, text="Отримати прогноз від усіх моделей", command=self.predict_cancellation, bg="#2b5c8f", fg="white", font=("Arial", 11, "bold"))
        btn_predict.pack(pady=15)

        # Поле виведення результатів
        self.txt_output = tk.Text(root, height=12, width=70)
        self.txt_output.pack(pady=5)

        # Кнопка збереження звіту
        btn_save = tk.Button(root, text="Зберегти текстовий звіт (.txt)", command=self.save_txt_report, bg="#4Caf50", fg="white")
        btn_save.pack(pady=5)

    def predict_cancellation(self):
        try:
            lead_time = float(self.entry_lead_time.get())
            adr = float(self.entry_adr.get())
            adults = float(self.entry_adults.get())
            prev_canc = float(self.entry_prev_cancellations.get())
            hotel = self.combo_hotel.get()
            deposit = self.combo_deposit.get()

            # Формування тестового структурованого запису з повним набором необхідних колонок
            input_df = pd.DataFrame([{
                'hotel': hotel, 'lead_time': lead_time, 'arrival_date_year': 2026,
                'arrival_date_month': 'July', 'arrival_date_week_number': 27,
                'arrival_date_day_of_month': 15, 'stays_in_weekend_nights': 1,
                'stays_in_week_nights': 2, 'adults': adults, 'children': 0, 'babies': 0,
                'meal': 'BB', 'country': 'PRT', 'market_segment': 'Online TA',
                'distribution_channel': 'TA/TO', 'is_repeated_guest': 0,
                'previous_cancellations': prev_canc, 'previous_bookings_not_canceled': 0,
                'reserved_room_type': 'A', 'assigned_room_type': 'A',
                'booking_changes': 0, 'deposit_type': deposit, 'agent': 9.0,
                'company': 0.0, 'days_in_waiting_list': 0, 'customer_type': 'Transient',
                'adr': adr, 'required_car_parking_spaces': 0, 'total_of_special_requests': 0
            }])

            self.txt_output.delete("1.0", tk.END)
            self.txt_output.insert(tk.END, f"РЕЗУЛЬТАТИ ПРОГНОЗУВАННЯ СКАСУВАННЯ ({datetime.now().strftime('%d.%m.%Y %H:%M')})\n")
            self.txt_output.insert(tk.END, "="*60 + "\n\n")

            if not self.models:
                self.txt_output.insert(tk.END, "Помилка: Не знайдено збережених моделей (.pkl).\nЗапустіть спочатку main.py!")
                return

            for name, model in self.models.items():
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
                status = "СКАСОВАНО (1)" if pred == 1 else "ПІДТВЕРДЖЕНО (0)"
                proba_str = f" | Ймовірність скасування: {proba*100:.1f}%" if proba is not None else ""
                
                self.txt_output.insert(tk.END, f"• {name}: {status}{proba_str}\n")

        except Exception as err:
            messagebox.showerror("Помилка даних", f"Перевірте правильностей введених числових значень:\n{err}")

    def save_txt_report(self):
        content = self.txt_output.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Увага", "Спочатку виконайне прогноз!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Збережено", f"Звіт успішно збережено у файл:\n{file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HotelBookingApp(root)
    root.mainloop()