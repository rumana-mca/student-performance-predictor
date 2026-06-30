import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# store history
prediction_history = []

# load dataset
df = pd.read_csv("student_data.csv")

X = df[[
    "StudyHours",
    "Attendance",
    "SleepHours",
    "PreviousMarks",
    "AssignmentsScore",
    "Participation"
]]

y = df["FinalMarks"]

# train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X, y)


def predict_marks():
    try:
        sh = float(entry1.get())
        att = float(entry2.get())
        sl = float(entry3.get())
        pm = float(entry4.get())
        ascore = float(entry5.get())
        part = float(entry6.get())

        if sh < 0 or att < 0 or sl < 0 or pm < 0 or ascore < 0 or part < 0:
            label_result.config(text="Enter positive values only!", fg="#ff4d4d")
            return

        new_data = pd.DataFrame(
            [[sh, att, sl, pm, ascore, part]],
            columns=[
                "StudyHours",
                "Attendance",
                "SleepHours",
                "PreviousMarks",
                "AssignmentsScore",
                "Participation"
            ]
        )

        result = model.predict(new_data)
        predicted = result[0]
        predicted = max(0, min(predicted, 100))

        if predicted >= 80:
            performance = "🟢 Excellent"
            suggestion = "Keep up the outstanding performance."
        elif predicted >= 60:
            performance = "🟡 Good"
            suggestion = "Consistent effort can improve results."
        elif predicted >= 40:
            performance = "🟠 Average"
            suggestion = "Focus more on assignments and study hours."
        else:
            performance = "🔴 Needs Improvement"
            suggestion = "Increase study time and attendance."

        label_result.config(
                 text=
                f"🎯 PredictedMarks : {predicted:.2f}/100\n\n"
                f"📈 Performance : {performance}\n\n"
                f"💡 Recommendation :\n{suggestion}",
                fg="#00ffcc"
        )

        history_entry = (
            f"Study:{sh}, Attend:{att}, Sleep:{sl}, Prev:{pm}, "
            f"Assign:{ascore}, Part:{part} -> {predicted:.2f}"
        )
        prediction_history.append(history_entry)

        history_box.delete("1.0", tk.END)
        for item in prediction_history:
            history_box.insert(tk.END, item + "\n")

    except Exception as e:
        label_result.config(text="Invalid input!", fg="#ff4d4d")
        print(e)


def clear_history():
    prediction_history.clear()
    history_box.delete("1.0", tk.END)


def export_history():
    if not prediction_history:
        label_result.config(text="No data to export!", fg="#ff4d4d")
        return

    with open("prediction_history.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "StudyHours", "Attendance", "SleepHours",
            "PreviousMarks", "AssignmentsScore",
            "Participation", "PredictedMarks"
        ])

        for item in prediction_history:
            parts = item.replace("Study:", "").replace("Attend:", "") \
                        .replace("Sleep:", "").replace("Prev:", "") \
                        .replace("Assign:", "").replace("Part:", "") \
                        .split(" -> ")

            values = parts[0].split(", ")
            predicted = parts[1]

            writer.writerow(values + [predicted])

    label_result.config(text="History exported successfully!", fg="#22c55e")

def show_performance():
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor()
    }

    best_model = ""
    best_r2 = -999

    result_text = "📊 MODEL PERFORMANCE SUMMARY\n\n"

    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items(): # Note: match your exact loop header here
        # 2. Fit ONLY on training data
        model.fit(X_train, y_train)
        
        # 3. Predict ONLY on test data
        y_pred = model.predict(X_test)
        
        # 4. Calculate metrics using the test variables
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_model = name

        result_text += (
            f"🔹 {name}\n"
            f"   MAE : {mae:.2f}\n"
            f"   R² Score : {r2:.2f}\n\n"
        )

    result_text += (
        f"🏆 Best Model : {best_model}\n"
        f"⭐ Highest R² Score : {best_r2:.2f}"
    )

    performance_label.config(
        text=result_text,
        fg="#22c55e",
        justify="left"
    )

def show_graph():
    for widget in graph_frame.winfo_children():
        widget.destroy()

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor()
    }

    model_names = []
    mae_scores = []

    for name, model in models.items():
        model.fit(X, y)

        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)

        model_names.append(name)
        mae_scores.append(mae)

    fig = plt.Figure(figsize=(6,3), dpi=100)

    ax = fig.add_subplot(111)

    ax.bar(model_names, mae_scores)

    ax.set_title("Model Comparison (MAE)")
    ax.set_ylabel("MAE")

    canvas_graph = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_graph.draw()
    canvas_graph.get_tk_widget().pack()

# GUI
root = tk.Tk()
root.title("🎓 Student Performance Prediction System")
root.geometry("600x900")

root.minsize(600, 700)
root.config(bg="#0f172a")

canvas = tk.Canvas(root, bg="#0f172a", highlightthickness=0)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)

scrollable_frame = tk.Frame(canvas, bg="#0f172a")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=580)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# title
tk.Label(
    scrollable_frame,
    text="🎓 Student Performance Prediction System",
    bg="#0f172a",
    fg="#38bdf8",
    font=("Segoe UI", 18, "bold")
).pack(pady=15)

tk.Label(
    scrollable_frame,
    text="Machine Learning Dashboard with Model Comparison",
    bg="#0f172a",
    fg="#94a3b8",
    font=("Segoe UI", 10, "italic")
).pack(pady=(0,10))

tk.Label(
    scrollable_frame,
    text="Enter all marks and percentages between 0–100",
    bg="#0f172a",
    fg="#94a3b8",
    font=("Segoe UI", 9)
).pack(pady=(0,15))

# frame
frame = tk.Frame(
    scrollable_frame,
    bg="#1e293b",
    padx=20,
    pady=20
)
frame.pack(pady=15, padx=20)

labels = [
    "Study Hours(per day)",
    "Attendance (%)",
    "Sleep Hours(per day)",
    "Previous Marks (%))",
    "Assignments Score (%)",
    "Participation (%)"
]

entries = []

for i, text in enumerate(labels):
    tk.Label(
        frame,
        text=text,
        bg="#0f172a",
        fg="white",
        font=("Segoe UI", 11)
    ).grid(row=i, column=0, padx=10, pady=6, sticky="w")

    entry = tk.Entry(
    frame,
    font=("Segoe UI", 11),
    width=22,
    bd=2,
    relief="groove"
)
    entry.grid(row=i, column=1, padx=10, pady=6)
    entries.append(entry)

entry1, entry2, entry3, entry4, entry5, entry6 = entries

# predict button
tk.Button(
    scrollable_frame,
    text="🚀 Predict Marks",
    cursor="hand2",
    bg="#22c55e",
    fg="black",
    font=("Segoe UI", 12, "bold"),
    padx=15,
    pady=8,
    command=predict_marks
).pack(pady=15)

# result
label_result = tk.Label(
    scrollable_frame,
    text="Prediction Result Will Appear Here",
    bg="#0f172a",
    fg="#00ffcc",
    font=("Segoe UI", 13, "bold")
)

label_result.pack(pady=15)

tk.Label(
    scrollable_frame,
    text="━━━━━━━━━━━━━━━━━━━━",
    bg="#0f172a",
    fg="#475569",
    font=("Segoe UI", 12)
).pack(pady=10)

# history title
tk.Label(
    scrollable_frame,
    text="📜 Prediction History",
    bg="#0f172a",
    fg="#38bdf8",
    font=("Segoe UI", 12, "bold")
).pack(pady=10)

# history box
history_box = tk.Text(
    scrollable_frame,
    height=8,
    width=65,
    bg="#111827",
    fg="#f8fafc",
    font=("Consolas", 9),
    bd=3,
    relief="ridge",
    insertbackground="white"
)

# button frame (NEW)
button_frame = tk.Frame(scrollable_frame, bg="#0f172a")
button_frame.pack(pady=10)

# performance title
tk.Label(
    scrollable_frame,
    text="📊 Model Performance",
    bg="#0f172a",
    fg="#38bdf8",
    font=("Segoe UI", 12, "bold")
).pack(pady=10)

# performance label
performance_label = tk.Label(
    scrollable_frame,
    text="Click 'Show Performance' to view results",
    bg="#0f172a",
    fg="white",
    font=("Segoe UI", 10),
    justify="left"
)
performance_label.pack()

tk.Button(
    scrollable_frame,
    text="Show Performance",
    cursor="hand2",
    bg="#f59e0b",
    fg="black",
    font=("Segoe UI", 10, "bold"),
    command=show_performance
).pack(pady=10)
tk.Label(
    scrollable_frame,
    text="📊 Graph Analysis",
    bg="#0f172a",
    fg="#38bdf8",
    font=("Segoe UI", 12, "bold")
).pack(pady=10)

graph_frame = tk.Frame(
    scrollable_frame,
    bg="#1e293b",
    bd=2,
    relief="ridge"
)

graph_frame.pack(pady=10)

tk.Button(
    scrollable_frame,
    text="📈 Show Graph",
    cursor="hand2",
    bg="#10b981",
    fg="black",
    font=("Segoe UI", 10, "bold"),
    command=show_graph
).pack(pady=10)

# clear button
tk.Button(
    button_frame,
    text="🗑 Clear History",
    cursor="hand2",
    bg="#ef4444",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    padx=10,
    pady=6,
    command=clear_history
).grid(row=0, column=0, padx=10)

# export button
tk.Button(
    button_frame,
    text="📁 Export to CSV",
    cursor="hand2",
    bg="#3b82f6",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    padx=10,
    pady=6,
    command=export_history
).grid(row=0, column=1, padx=10)

tk.Label(
    scrollable_frame,
    text="💻 Developed by Rumana Khan | MCA Major Project",
    bg="#0f172a",
    fg="#64748b",
    font=("Segoe UI", 9, "italic")
).pack(pady=15)

# run
root.mainloop()