# Import required libraries
import os
import simpy
import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Simulation Parameters
SIM_TIME = 24 * 60  # Total simulation time in minutes
CLOSING_TIMES = [20 * 60, 22 * 60, 24 * 60]  # 8 PM, 10 PM, 12 AM in minutes
HOLDING_BAY_COUNTS = [11, 15, 18, 22]  # Different numbers of holding bays to test
CONTRIBUTION_MARGIN = 600  # Contribution margin per completed procedure
HOLDING_BAY_COST_PER_HOUR = 10  # Cost per holding bay per hour
CANCELLATION_COST_PER_PROCEDURE = 600  # Cancellation cost per procedure
STAFF_COST_BASE = 100  # Example base staff cost per hour (adjust as needed)
STAFF_COST_OVERTIME = 150  # Example overtime staff cost per hour (adjust as needed)

# Arrival Rates (per minute)
ARRIVAL_RATES = {
    'CATH': [
        (6 * 60, 10 * 60, 10 / 60),
        (10 * 60, 14 * 60, 8 / 60),
        (14 * 60, 18 * 60, 5 / 60),
    ],
    'EP': [
        (6 * 60, 10 * 60, 15 / 60),
        (10 * 60, 14 * 60, 10 / 60),
        (14 * 60, 18 * 60, 8 / 60),
    ],
}

# Load and model input data
data = pd.read_csv('hospital-lab-scheduling-and-resource-allocation\Data set for Lab and recovery durations.csv')

# Fit lognormal distributions to lab and recovery times
cath_lab_times = data['CATH Lab time (in minutes)'].dropna()
cath_shape, cath_loc, cath_scale = lognorm.fit(cath_lab_times, floc=0)

ep_lab_times = data['EP Lab time (in minutes)'].dropna()
ep_shape, ep_loc, ep_scale = lognorm.fit(ep_lab_times, floc=0)

recovery_times = data['Recovery time (in minutes)'].dropna()
recovery_shape, recovery_loc, recovery_scale = lognorm.fit(recovery_times, floc=0)

# Functions to generate procedure and recovery times
def generate_procedure_time(patient_type):
    if patient_type == 'CATH':
        # Generate procedure time for CATH patients using fitted lognormal distribution
        return lognorm(cath_shape, loc=cath_loc, scale=cath_scale).rvs()
    elif patient_type == 'EP':
        # Generate procedure time for EP patients using fitted lognormal distribution
        return lognorm(ep_shape, loc=ep_loc, scale=ep_scale).rvs()

def generate_recovery_time():
    return lognorm(recovery_shape, loc=recovery_loc, scale=recovery_scale).rvs()

# Confidence Interval Calculation
def calculate_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_error = np.std(data, ddof=1) / np.sqrt(len(data))
    margin = norm.ppf((1 + confidence) / 2) * std_error
    return mean - margin, mean + margin

# Cost and Profit Calculation
def calculate_costs_and_profit(stats, num_holding_bays, closing_time):
    # Calculate Revenue
    revenue = stats['processed'] * CONTRIBUTION_MARGIN

    # Holding Bay Costs
    operational_hours = closing_time / 60  # Convert closing time to hours
    holding_bay_cost = num_holding_bays * operational_hours * HOLDING_BAY_COST_PER_HOUR

    # Cancellation Costs
    cancellation_cost = stats['cancelled'] * CANCELLATION_COST_PER_PROCEDURE

    # Staff Costs (simple ratio for illustration)
    base_staff_cost = STAFF_COST_BASE * operational_hours
    overtime_staff_cost = STAFF_COST_OVERTIME * max(0, (stats['arrivals'] - stats['processed']) / 5)  # Example overtime logic

    # Total Costs
    total_cost = holding_bay_cost + cancellation_cost + base_staff_cost + overtime_staff_cost

    # Profit Calculation
    profit = revenue - total_cost

    return revenue, holding_bay_cost, cancellation_cost, base_staff_cost, overtime_staff_cost, total_cost, profit

# Visualization
def plot_results(results_df):
    # Directory where plots will be saved
    output_directory = os.path.join(os.getcwd(), 'hospital-lab-scheduling-and-resource-allocation')
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

    # Plot 1: Profit vs Holding Bays and Closing Time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='Holding Bays', y='Profit', hue='Closing Time')
    plt.title('Profit by Holding Bays and Closing Time')
    plt.xlabel('Holding Bays')
    plt.ylabel('Profit ($)')
    plt.legend(title='Closing Time (Hours)')
    plt.tight_layout()
    profit_plot_path = os.path.join(output_directory, 'profit_vs_holding_bays.png')
    plt.savefig(profit_plot_path)
    print(f"Saved plot: {profit_plot_path}")
    plt.close()

    # Plot 2: Cost Breakdown by Holding Bays
    results_df_melted = results_df.melt(
        id_vars=['Holding Bays', 'Closing Time'], 
        value_vars=['Holding Bay Cost', 'Cancellation Cost', 'Base Staff Cost', 'Overtime Staff Cost'],
        var_name='Cost Type', value_name='Cost ($)'
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df_melted, x='Holding Bays', y='Cost ($)', hue='Cost Type')
    plt.title('Cost Breakdown by Holding Bays')
    plt.xlabel('Holding Bays')
    plt.ylabel('Cost ($)')
    plt.legend(title='Cost Type')
    plt.tight_layout()
    cost_plot_path = os.path.join(output_directory, 'cost_breakdown_by_holding_bays.png')
    plt.savefig(cost_plot_path)
    print(f"Saved plot: {cost_plot_path}")
    plt.close()

# Additional Visualization: Plot Fitted Distributions
def plot_fitted_distributions(data, column_name, distribution_params, title, filename):
    shape, loc, scale = distribution_params
    output_directory = os.path.join(os.getcwd(), 'hospital-lab-scheduling-and-resource-allocation')
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

    # Plot the histogram and fitted distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=30, kde=False, stat='density', color='cyan', label='Data')
    x = np.linspace(min(data), max(data), 100)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)
    plt.plot(x, pdf, color='blue', label='Fitted Lognormal')
    plt.title(title)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_directory, filename)
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

# Generate and save plots for CATH, EP, and Recovery times
plot_fitted_distributions(
    data=cath_lab_times,
    column_name='CATH Lab time (in minutes)',
    distribution_params=(cath_shape, cath_loc, cath_scale),
    title='CATH Lab Times Distribution',
    filename='cath_lab_times_distribution.png'
)

plot_fitted_distributions(
    data=ep_lab_times,
    column_name='EP Lab time (in minutes)',
    distribution_params=(ep_shape, ep_loc, ep_scale),
    title='EP Lab Times Distribution',
    filename='ep_lab_times_distribution.png'
)

plot_fitted_distributions(
    data=recovery_times,
    column_name='Recovery time (in minutes)',
    distribution_params=(recovery_shape, recovery_loc, recovery_scale),
    title='Recovery Times Distribution',
    filename='recovery_times_distribution.png'
)


# Hospital Class
class Hospital:
    def __init__(self, env, num_holding_bays, closing_time):
        self.env = env
        self.holding_bays = simpy.PriorityResource(env, capacity=num_holding_bays)
        self.lab_units = simpy.Resource(env, capacity=12)
        self.closing_time = closing_time
        self.holding_bays_open = True
        self.stats = {
            'arrivals': 0,
            'processed': 0,
            'reneged': 0,
            'cancelled': 0,
            'transferred': 0,
        }
        env.process(self.close_holding_bays())

    def close_holding_bays(self):
        yield self.env.timeout(self.closing_time)
        self.holding_bays_open = False

# Patient Process
def patient(env, name, patient_type, hospital):
    arrival_time = env.now
    hospital.stats['arrivals'] += 1

    if not hospital.holding_bays_open:
        hospital.stats['cancelled'] += 1
        return

    with hospital.holding_bays.request(priority=1) as request:
        result = yield request | env.timeout(60)
        if request in result:
            preparation_time = np.random.exponential(20)
            yield env.timeout(preparation_time)

            with hospital.lab_units.request() as lab_request:
                yield lab_request
                hospital.holding_bays.release(request)

                procedure_duration = generate_procedure_time(patient_type)
                yield env.timeout(procedure_duration)

                if hospital.holding_bays_open:
                    with hospital.holding_bays.request(priority=0) as recovery_request:
                        yield recovery_request
                        hospital.lab_units.release(lab_request)
                        recovery_duration = generate_recovery_time()
                        yield env.timeout(recovery_duration)
                        hospital.holding_bays.release(recovery_request)
                        hospital.stats['processed'] += 1
                else:
                    hospital.lab_units.release(lab_request)
                    hospital.stats['transferred'] += 1
                    hospital.stats['processed'] += 1
        else:
            hospital.stats['reneged'] += 1

# Simulation
def run_simulation(num_holding_bays, closing_time):
    env = simpy.Environment()
    hospital = Hospital(env, num_holding_bays, closing_time)
    env.process(arrival_process(env, 'CATH', hospital))
    env.process(arrival_process(env, 'EP', hospital))
    env.run(until=SIM_TIME)
    return hospital.stats

# Arrival Process
def arrival_process(env, patient_type, hospital):
    while True:
        current_time = env.now
        arrival_rate = get_arrival_rate(patient_type, current_time)
        if arrival_rate == 0 or current_time >= hospital.closing_time:
            yield env.timeout(1)
            continue
        interarrival_time = random.expovariate(arrival_rate)
        yield env.timeout(interarrival_time)
        if env.now >= hospital.closing_time:
            break
        patient_id = f"{patient_type}_{hospital.stats['arrivals'] + 1}"
        env.process(patient(env, patient_id, patient_type, hospital))

# Get Arrival Rate
def get_arrival_rate(patient_type, current_time):
    for start, end, rate in ARRIVAL_RATES[patient_type]:
        if start <= current_time < end:
            return rate
    return 0

# Main Simulation Loop
results = []
for num_bays in HOLDING_BAY_COUNTS:
    for closing in CLOSING_TIMES:
        stats = run_simulation(num_bays, closing)
        revenue, holding_bay_cost, cancellation_cost, base_staff_cost, overtime_staff_cost, total_cost, profit = calculate_costs_and_profit(stats, num_bays, closing)

        # Append results to the DataFrame
        results.append({
            'Holding Bays': num_bays,
            'Closing Time': int(closing / 60),
            'Total Arrivals': stats['arrivals'],
            'Processed': stats['processed'],
            'Reneged': stats['reneged'],
            'Cancelled': stats['cancelled'],
            'Transferred': stats['transferred'],
            'Revenue': revenue,
            'Holding Bay Cost': holding_bay_cost,
            'Cancellation Cost': cancellation_cost,
            'Base Staff Cost': base_staff_cost,
            'Overtime Staff Cost': overtime_staff_cost,
            'Total Cost': total_cost,
            'Profit': profit,
        })

        # Print detailed simulation breakdown
        print(f"Simulation with {num_bays} holding bays and closing time at {int(closing / 60)}:00")
        print(f"Total Arrivals: {stats['arrivals']}")
        print(f"Processed: {stats['processed']}")
        print(f"Reneged: {stats['reneged']}")
        print(f"Cancelled: {stats['cancelled']}")
        print(f"Transferred: {stats['transferred']}")
        print(f"Revenue: ${revenue:.2f}")
        print(f"Holding Bay Cost: ${holding_bay_cost:.2f}")
        print(f"Cancellation Cost: ${cancellation_cost:.2f}")
        print(f"Base Staff Cost: ${base_staff_cost:.2f}")
        print(f"Overtime Staff Cost: ${overtime_staff_cost:.2f}")
        print(f"Total Cost: ${total_cost:.2f}")
        print(f"Profit: ${profit:.2f}")
        print("-" * 50)

# Convert results to DataFrame for further analysis
results_df = pd.DataFrame(results)
print(results_df)
plot_results(results_df)