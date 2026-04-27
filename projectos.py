import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


st.set_page_config(page_title="AI CPU Scheduler", layout="wide")

st.title("🧠 AI-Based Predictive CPU Scheduler")
st.markdown(
    "Uses **Linear Regression** to predict CPU burst times from historical patterns. "
    "Compares **FCFS, SJF, Round Robin, and AI-SJF** on waiting time, "
    "turnaround time, and response time under varying workloads."
)

st.sidebar.header("⚙️ Configuration")

num_processes = st.sidebar.slider("Number of Processes", 4, 12, 8)
workload_type = st.sidebar.selectbox(
    "Workload Type",
    ["Mixed", "CPU-bound", "I/O-bound", "Bursty"],
    help="Simulates different real-world scheduling scenarios"
)
quantum = st.sidebar.slider("Round Robin Quantum (ms)", 1, 10, 4)
noise_level = st.sidebar.slider("ML Prediction Noise", 0.5, 3.0, 1.5, step=0.5,
    help="Controls how accurate the ML burst prediction is")
run = st.sidebar.button("🚀 Run Simulation", use_container_width=True)


def generate_workload(workload_type, n, seed=42):
    rng = np.random.default_rng(seed)
    
    if workload_type == "CPU-bound":
        pattern = [18, 22, 20, 25, 17, 24, 19, 23, 21, 20, 18, 22]
    elif workload_type == "I/O-bound":
        pattern = [3, 5, 4, 2, 6, 3, 5, 4, 3, 5, 4, 6]
    elif workload_type == "Bursty":
        pattern = [4, 5, 30, 4, 6, 28, 5, 4, 29, 5, 4, 27]
    else:  
        pattern = [12, 15, 13, 8, 20, 18, 22, 10, 14, 16, 11, 19]
    

    bursts = np.array([pattern[i % len(pattern)] + rng.normal(0, 1.5) for i in range(n)])
    bursts = np.clip(bursts, 1, 50).round().astype(float)
    
    arrivals = np.round(np.arange(n) * rng.uniform(0, 2, n)).astype(int)
    return bursts, arrivals

# ML MODEL
def generate_historical_data(workload_type, seed=0):
    rng = np.random.default_rng(seed)
    if workload_type == "CPU-bound":
        pattern = [18, 22, 20, 25, 17, 24, 19, 23]
    elif workload_type == "I/O-bound":
        pattern = [3, 5, 4, 2, 6, 3, 5, 4]
    elif workload_type == "Bursty":
        pattern = [4, 5, 30, 4, 6, 28, 5, 4]
    else:
        pattern = [12, 15, 13, 8, 20, 18, 22, 10]
    data = []
    for _ in range(100):
        for val in pattern:
            data.append(val + rng.normal(0, 2))
    return np.array(data)

def train_ml_model(data):
    X, y = [], []
    for i in range(len(data) - 4):
        X.append(data[i:i+3])
        y.append(data[i+3])
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_bursts(model, actual_bursts, noise_level):
    rng = np.random.default_rng(99)
    predicted = []
    for b in actual_bursts:
        history = np.array([b * 0.9, b, b * 1.1]).reshape(1, -1)
        pred = model.predict(history)[0]
        pred += rng.normal(0, noise_level)
        predicted.append(max(1.0, round(pred, 1)))
    return predicted

def fcfs(processes):
    result = []
    time = 0
    for p in processes:
        start = max(time, p['arrival'])
        finish = start + p['burst']
        time = finish
        result.append({**p,
            'start': start, 'finish': finish,
            'wait': start - p['arrival'],
            'tat': finish - p['arrival'],
            'response': start - p['arrival']
        })
    return result

def sjf(processes, use_predicted=False):
    key = 'predicted' if use_predicted else 'burst'
    sorted_p = sorted(processes, key=lambda x: x[key])
    return fcfs(sorted_p)

def round_robin(processes, quantum):
    queue = [{**p, 'remaining': p['burst'], 'first_start': -1} for p in processes]
    time = 0
    slices = {p['pid']: [] for p in processes}
    first_start = {p['pid']: -1 for p in processes}

    while any(p['remaining'] > 0 for p in queue):
        progressed = False
        for p in queue:
            if p['remaining'] <= 0:
                continue
            progressed = True
            start = max(time, p['arrival'])
            if first_start[p['pid']] == -1:
                first_start[p['pid']] = start
            run_time = min(quantum, p['remaining'])
            slices[p['pid']].append((start, start + run_time))
            time = start + run_time
            p['remaining'] -= run_time
        if not progressed:
            break

    result = []
    for p in queue:
        s = slices[p['pid']]
        if not s:
            continue
        finish = max(e for _, e in s)
        wait = max(0, finish - p['arrival'] - p['burst'])
        resp = max(0, first_start[p['pid']] - p['arrival'])
        result.append({**p,
            'start': first_start[p['pid']], 'finish': finish,
            'wait': wait, 'tat': finish - p['arrival'],
            'response': resp, 'slices': s
        })
    return result
def compute_accuracy(actual, predicted):
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    rel = sum(e / a for e, a in zip(errors, actual) if a > 0) / len(actual)
    return round((1 - rel) * 100, 2)

# GANTT CHART
def draw_gantt(result, algo_name, ax, use_slices=False):
    colors_exec = '#378ADD'
    colors_wait = '#E8E8E8'
    max_time = max(p['finish'] for p in result)

    for i, p in enumerate(result):
        y = i
        if use_slices and 'slices' in p:
            # draw wait background
            ax.barh(y, max_time, left=0, height=0.5, color=colors_wait, edgecolor='none')
            for (s, e) in p['slices']:
                ax.barh(y, e - s, left=s, height=0.5, color=colors_exec, edgecolor='white', linewidth=0.4)
                ax.text(s + (e - s) / 2, y, str(e - s), ha='center', va='center', fontsize=7, color='white')
        else:
            # wait bar
            ax.barh(y, p['wait'], left=p['arrival'], height=0.5, color=colors_wait,
                    edgecolor='#cccccc', linewidth=0.4)
            # burst bar
            ax.barh(y, p['burst'], left=p['start'], height=0.5, color=colors_exec,
                    edgecolor='white', linewidth=0.4)
            ax.text(p['start'] + p['burst'] / 2, y, str(p['burst']),
                    ha='center', va='center', fontsize=7, color='white')
            
    ax.set_yticks(range(len(result)))
    ax.set_yticklabels([p['pid'] for p in result], fontsize=9)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_title(f"Gantt — {algo_name}", fontsize=10, fontweight='bold')
    ax.set_xlim(0, max_time + 1)
    exec_patch = mpatches.Patch(color=colors_exec, label='Executing')
    wait_patch = mpatches.Patch(color=colors_wait, label='Waiting', edgecolor='#ccc')
    ax.legend(handles=[exec_patch, wait_patch], fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# MAIN SIMULATION
if run:
    actual_bursts, arrivals = generate_workload(workload_type, num_processes)
    hist_data = generate_historical_data(workload_type)
    model = train_ml_model(hist_data)
    predicted_bursts = predict_bursts(model, actual_bursts, noise_level)

    processes = [
        {'pid': f'P{i}', 'burst': int(actual_bursts[i]),
         'predicted': predicted_bursts[i], 'arrival': int(arrivals[i])}
        for i in range(num_processes)
    ]

    # Run all algorithms
    res_fcfs = fcfs(list(processes))
    res_sjf  = sjf(list(processes), use_predicted=False)
    res_rr   = round_robin(list(processes), quantum)
    res_ai   = sjf(list(processes), use_predicted=True)

    accuracy = compute_accuracy(actual_bursts, predicted_bursts)

    def mean(lst, key):
        return round(sum(p[key] for p in lst) / len(lst), 2)

    st.markdown("---")
    st.subheader("📊 Performance Metrics (AI-SJF)")
    c1, c2, c3, c4, c5 = st.columns(5)
    avg_w_fcfs = mean(res_fcfs, 'wait')
    avg_w_ai   = mean(res_ai, 'wait')
    pct = round((avg_w_ai - avg_w_fcfs) / avg_w_fcfs * 100, 1) if avg_w_fcfs > 0 else 0

    c1.metric("FCFS Avg Wait",      f"{avg_w_fcfs} ms")
    c2.metric("AI-SJF Avg Wait",    f"{avg_w_ai} ms",          delta=f"{pct}%", delta_color="inverse")
    c3.metric("Avg Turnaround",     f"{mean(res_ai,'tat')} ms")
    c4.metric("Avg Response Time",  f"{mean(res_ai,'response')} ms")
    c5.metric("ML Accuracy",        f"{accuracy}%")

    # GANTT CHARTS
    st.markdown("---")
    st.subheader("📅 Gantt Charts")

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle("CPU Scheduling — Gantt Charts", fontsize=12, fontweight='bold')
    draw_gantt(res_fcfs, "FCFS",              axes[0][0])
    draw_gantt(res_sjf,  "SJF (True)",        axes[0][1])
    draw_gantt(res_rr,   "Round Robin",       axes[1][0], use_slices=True)
    draw_gantt(res_ai,   "AI-SJF (Predicted)",axes[1][1])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    #  COMPARISON CHARTS 
    st.markdown("---")
    st.subheader("📈 Algorithm Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        pids = [p['pid'] for p in processes]
        x = np.arange(len(pids))
        ax1.bar(x - 0.2, actual_bursts, 0.4, label='Actual', color='#378ADD')
        ax1.bar(x + 0.2, predicted_bursts, 0.4, label='Predicted (ML)', color='#EF9F27')
        ax1.set_xticks(x); ax1.set_xticklabels(pids, fontsize=9)
        ax1.set_xlabel("Process"); ax1.set_ylabel("Burst Time (ms)")
        ax1.set_title("Actual vs Predicted Burst Time", fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        st.pyplot(fig1); plt.close()

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        algos = ['FCFS', 'SJF', 'Round Robin', 'AI-SJF']
        wait_vals = [mean(res_fcfs,'wait'), mean(res_sjf,'wait'), mean(res_rr,'wait'), mean(res_ai,'wait')]
        bar_colors = ['#888780', '#378ADD', '#1D9E75', '#EF9F27']
        ax2.bar(algos, wait_vals, color=bar_colors)
        for i, v in enumerate(wait_vals):
            ax2.text(i, v + 0.2, str(v), ha='center', fontsize=9)
        ax2.set_ylabel("Avg Waiting Time (ms)")
        ax2.set_title("Avg Waiting Time by Algorithm", fontweight='bold')
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        st.pyplot(fig2); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fcfs_tat = [p['tat'] for p in sorted(res_fcfs, key=lambda x: x['pid'])]
        ai_tat   = [p['tat'] for p in sorted(res_ai,   key=lambda x: x['pid'])]
        ax3.bar(x - 0.2, fcfs_tat, 0.4, label='FCFS',    color='#888780')
        ax3.bar(x + 0.2, ai_tat,   0.4, label='AI-SJF',  color='#EF9F27')
        ax3.set_xticks(x); ax3.set_xticklabels(pids, fontsize=9)
        ax3.set_xlabel("Process"); ax3.set_ylabel("Turnaround Time (ms)")
        ax3.set_title("Turnaround Time: FCFS vs AI-SJF", fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
        st.pyplot(fig3); plt.close()

    with col4:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        resp_vals = [mean(res_fcfs,'response'), mean(res_sjf,'response'),
                     mean(res_rr,'response'), mean(res_ai,'response')]
        ax4.bar(algos, resp_vals, color=bar_colors)
        for i, v in enumerate(resp_vals):
            ax4.text(i, v + 0.1, str(v), ha='center', fontsize=9)
        ax4.set_ylabel("Avg Response Time (ms)")
        ax4.set_title("Avg Response Time by Algorithm", fontweight='bold')
        ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
        st.pyplot(fig4); plt.close()

    #PROCESS TABLE 
    st.markdown("---")
    st.subheader(" Detailed Process Table — AI-SJF (Optimized Queue)")
    df_ai = pd.DataFrame(res_ai)[['pid','burst','predicted','arrival','start','finish','wait','tat','response']]
    df_ai.columns = ['PID','Burst (Actual)','Burst (Predicted)','Arrival','Start','Finish','Wait','Turnaround','Response']
    st.dataframe(df_ai.reset_index(drop=True), use_container_width=True)

    # SUMMARY TABLE
    st.markdown("---")
    st.subheader(" Algorithm Summary")
    base_wait = mean(res_fcfs, 'wait')
    summary = []
    for name, res in [('FCFS', res_fcfs), ('SJF (True)', res_sjf),
                       ('Round Robin', res_rr), ('AI-SJF (Predicted)', res_ai)]:
        w = mean(res, 'wait')
        t = mean(res, 'tat')
        r = mean(res, 'response')
        imp = '—' if name == 'FCFS' else f"{round((w - base_wait)/base_wait*100, 1)}%"
        summary.append({'Algorithm': name, 'Avg Wait (ms)': w,
                         'Avg Turnaround (ms)': t, 'Avg Response (ms)': r,
                         'Improvement vs FCFS': imp})
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


else:
    st.info(" Configure the settings in the sidebar and click **Run Simulation** to start.")