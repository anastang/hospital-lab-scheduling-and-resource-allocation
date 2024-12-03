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
cath_lab_times = data['CATH Lab time (in minutes)'].dropna()
cath_shape, cath_loc, cath_scale = lognorm.fit(cath_lab_times, floc=0)

ep_lab_times = data['EP Lab time (in minutes)'].dropna()
ep_shape, ep_loc, ep_scale = lognorm.fit(ep_lab_times, floc=0)

recovery_times = data['Recovery time (in minutes)'].dropna()
recovery_shape, recovery_loc, recovery_scale = lognorm.fit(recovery_times, floc=0)

# Functions to generate procedure and recovery times
def generate_procedure_time(patient_type):
    if patient_type == 'CATH':
        return lognorm(cath_shape, loc=cath_loc, scale=cath_scale).rvs()
    elif patient_type == 'EP':
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

# Main Simulation Loop with Detailed Output
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



# import simpy
# import numpy as np
# import pandas as pd
# from scipy.stats import lognorm, norm
# import random
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set random seed for reproducibility
# random.seed(42)
# np.random.seed(42)

# # Simulation Parameters
# SIM_TIME = 24 * 60  # Total simulation time in minutes
# CLOSING_TIMES = [20 * 60, 22 * 60, 24 * 60]  # 8 PM, 10 PM, 12 AM in minutes
# HOLDING_BAY_COUNTS = [11, 15, 18, 22]  # Different numbers of holding bays to test

# # Arrival Rates (per minute)
# ARRIVAL_RATES = {
#     'CATH': [
#         (6 * 60, 10 * 60, 10 / 60),
#         (10 * 60, 14 * 60, 8 / 60),
#         (14 * 60, 18 * 60, 5 / 60),
#     ],
#     'EP': [
#         (6 * 60, 10 * 60, 15 / 60),
#         (10 * 60, 14 * 60, 10 / 60),
#         (14 * 60, 18 * 60, 8 / 60),
#     ],
# }

# # Load and model input data
# data = pd.read_csv('hospital-lab-scheduling-and-resource-allocation\Data set for Lab and recovery durations.csv')
# cath_lab_times = data['CATH Lab time (in minutes)'].dropna()
# cath_shape, cath_loc, cath_scale = lognorm.fit(cath_lab_times, floc=0)

# ep_lab_times = data['EP Lab time (in minutes)'].dropna()
# ep_shape, ep_loc, ep_scale = lognorm.fit(ep_lab_times, floc=0)

# recovery_times = data['Recovery time (in minutes)'].dropna()
# recovery_shape, recovery_loc, recovery_scale = lognorm.fit(recovery_times, floc=0)

# # Functions to generate procedure and recovery times
# def generate_procedure_time(patient_type):
#     if patient_type == 'CATH':
#         return lognorm(cath_shape, loc=cath_loc, scale=cath_scale).rvs()
#     elif patient_type == 'EP':
#         return lognorm(ep_shape, loc=ep_loc, scale=ep_scale).rvs()

# def generate_recovery_time():
#     return lognorm(recovery_shape, loc=recovery_loc, scale=recovery_scale).rvs()

# # Confidence Interval Calculation
# def calculate_confidence_interval(data, confidence=0.95):
#     mean = np.mean(data)
#     std_error = np.std(data, ddof=1) / np.sqrt(len(data))
#     margin = norm.ppf((1 + confidence) / 2) * std_error
#     return mean - margin, mean + margin

# # Cost Calculation
# def calculate_costs(stats, num_holding_bays):
#     cancellation_cost = stats['reneged'] * 600  # Assuming $600 per cancelled patient
#     empty_bay_cost = (SIM_TIME - stats['holding_bay_idle_time']) / 60 * 10 * num_holding_bays  # $10/hour
#     total_cost = cancellation_cost + empty_bay_cost
#     return cancellation_cost, empty_bay_cost, total_cost

# # Visualization
# def plot_results(results_df):
#     sns.lineplot(data=results_df, x='Holding Bays', y='Processed', hue='Closing Time')
#     plt.title('Processed Patients by Holding Bays and Closing Time')
#     plt.show()

#     sns.barplot(data=results_df, x='Holding Bays', y='Reneged', hue='Closing Time')
#     plt.title('Reneged Patients by Holding Bays and Closing Time')
#     plt.show()

# # Hospital Class
# class Hospital:
#     def __init__(self, env, num_holding_bays, closing_time):
#         self.env = env
#         self.holding_bays = simpy.PriorityResource(env, capacity=num_holding_bays)
#         self.lab_units = simpy.Resource(env, capacity=12)
#         self.closing_time = closing_time
#         self.holding_bays_open = True
#         self.stats = {
#             'arrivals': 0,
#             'processed': 0,
#             'reneged': 0,
#             'cancelled': 0,
#             'transferred': 0,
#             'holding_bay_idle_time': 0,
#             'lab_unit_idle_time': 0,
#         }
#         env.process(self.close_holding_bays())

#     def close_holding_bays(self):
#         yield self.env.timeout(self.closing_time)
#         self.holding_bays_open = False

# # Patient Process
# def patient(env, name, patient_type, hospital):
#     arrival_time = env.now
#     hospital.stats['arrivals'] += 1

#     if not hospital.holding_bays_open:
#         hospital.stats['cancelled'] += 1
#         return

#     with hospital.holding_bays.request(priority=1) as request:
#         result = yield request | env.timeout(60)
#         if request in result:
#             preparation_time = np.random.exponential(20)
#             hospital.stats['holding_bay_idle_time'] += preparation_time
#             yield env.timeout(preparation_time)

#             with hospital.lab_units.request() as lab_request:
#                 yield lab_request
#                 hospital.holding_bays.release(request)

#                 procedure_duration = generate_procedure_time(patient_type)
#                 hospital.stats['lab_unit_idle_time'] += procedure_duration
#                 yield env.timeout(procedure_duration)

#                 if hospital.holding_bays_open:
#                     with hospital.holding_bays.request(priority=0) as recovery_request:
#                         yield recovery_request
#                         hospital.lab_units.release(lab_request)
#                         recovery_duration = generate_recovery_time()
#                         yield env.timeout(recovery_duration)
#                         hospital.holding_bays.release(recovery_request)
#                         hospital.stats['processed'] += 1
#                 else:
#                     hospital.lab_units.release(lab_request)
#                     hospital.stats['transferred'] += 1
#                     hospital.stats['processed'] += 1
#         else:
#             hospital.stats['reneged'] += 1

# # Simulation
# def run_simulation(num_holding_bays, closing_time):
#     env = simpy.Environment()
#     hospital = Hospital(env, num_holding_bays, closing_time)
#     env.process(arrival_process(env, 'CATH', hospital))
#     env.process(arrival_process(env, 'EP', hospital))
#     env.run(until=SIM_TIME)
#     return hospital.stats

# # Arrival Process
# def arrival_process(env, patient_type, hospital):
#     while True:
#         current_time = env.now
#         arrival_rate = get_arrival_rate(patient_type, current_time)
#         if arrival_rate == 0 or current_time >= hospital.closing_time:
#             yield env.timeout(1)
#             continue
#         interarrival_time = random.expovariate(arrival_rate)
#         yield env.timeout(interarrival_time)
#         if env.now >= hospital.closing_time:
#             break
#         patient_id = f"{patient_type}_{hospital.stats['arrivals'] + 1}"
#         env.process(patient(env, patient_id, patient_type, hospital))

# # Get Arrival Rate
# def get_arrival_rate(patient_type, current_time):
#     for start, end, rate in ARRIVAL_RATES[patient_type]:
#         if start <= current_time < end:
#             return rate
#     return 0

# # Main Simulation Loop with Detailed Output
# results = []
# for num_bays in HOLDING_BAY_COUNTS:
#     for closing in CLOSING_TIMES:
#         stats = run_simulation(num_bays, closing)
#         cancellation_cost, empty_bay_cost, total_cost = calculate_costs(stats, num_bays)

#         # Append results to the DataFrame
#         results.append({
#             'Holding Bays': num_bays,
#             'Closing Time': int(closing / 60),
#             'Total Arrivals': stats['arrivals'],
#             'Processed': stats['processed'],
#             'Reneged': stats['reneged'],
#             'Cancelled': stats['cancelled'],
#             'Transferred': stats['transferred'],
#             'Cancellation Cost': cancellation_cost,
#             'Empty Bay Cost': empty_bay_cost,
#             'Total Cost': total_cost,
#         })

#         # Print detailed simulation breakdown
#         print(f"Simulation with {num_bays} holding bays and closing time at {int(closing / 60)}:00")
#         print(f"Total Arrivals: {stats['arrivals']}")
#         print(f"Processed: {stats['processed']}")
#         print(f"Reneged: {stats['reneged']}")
#         print(f"Cancelled: {stats['cancelled']}")
#         print(f"Transferred: {stats['transferred']}")
#         print(f"Cancellation Cost: ${cancellation_cost:.2f}")
#         print(f"Empty Bay Cost: ${empty_bay_cost:.2f}")
#         print(f"Total Cost: ${total_cost:.2f}")
#         print("-" * 50)

# # Convert results to DataFrame for further analysis
# results_df = pd.DataFrame(results)
# print(results_df)
# plot_results(results_df)


# import simpy
# import numpy as np
# import pandas as pd
# from scipy.stats import lognorm
# import random
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set random seed for reproducibility
# random.seed(42)
# np.random.seed(42)

# # Simulation Parameters
# SIM_TIME = 24 * 60  # Total simulation time in minutes
# CLOSING_TIMES = [20 * 60, 22 * 60, 24 * 60]  # 8 PM, 10 PM, 12 AM in minutes
# HOLDING_BAY_COUNTS = [11, 15, 18, 22]  # Different numbers of holding bays to test

# # Arrival Rates (per minute)
# ARRIVAL_RATES = {
#     'CATH': [
#         (6 * 60, 10 * 60, 10 / 60),  # 6 AM to 10 AM
#         (10 * 60, 14 * 60, 8 / 60),  # 10 AM to 2 PM
#         (14 * 60, 18 * 60, 5 / 60),  # 2 PM to 6 PM
#     ],
#     'EP': [
#         (6 * 60, 10 * 60, 15 / 60),  # 6 AM to 10 AM
#         (10 * 60, 14 * 60, 10 / 60),  # 10 AM to 2 PM
#         (14 * 60, 18 * 60, 8 / 60),  # 2 PM to 6 PM
#     ],
# }

# # 1. Load and model input data
# data = pd.read_csv('hospital-lab-scheduling-and-resource-allocation\Data set for Lab and recovery durations.csv')

# # Fit distributions to CATH Lab time
# cath_lab_times = data['CATH Lab time (in minutes)'].dropna()
# cath_shape, cath_loc, cath_scale = lognorm.fit(cath_lab_times, floc=0)

# # Fit distributions to EP Lab time
# ep_lab_times = data['EP Lab time (in minutes)'].dropna()
# ep_shape, ep_loc, ep_scale = lognorm.fit(ep_lab_times, floc=0)

# # Fit distribution to Recovery time
# recovery_times = data['Recovery time (in minutes)'].dropna()
# recovery_shape, recovery_loc, recovery_scale = lognorm.fit(recovery_times, floc=0)

# def generate_procedure_time(patient_type):
#     """Generate a procedure time based on the patient type and fitted distributions."""
#     if patient_type == 'CATH':
#         return lognorm(cath_shape, loc=cath_loc, scale=cath_scale).rvs()
#     elif patient_type == 'EP':
#         return lognorm(ep_shape, loc=ep_loc, scale=ep_scale).rvs()
#     else:
#         raise ValueError("Unknown patient type")

# def generate_recovery_time():
#     """Generate a recovery time based on the fitted distribution."""
#     return lognorm(recovery_shape, loc=recovery_loc, scale=recovery_scale).rvs()

# # Optional: Plot the fitted distributions for verification
# def plot_fitted_distributions():
#     # Plot CATH Lab times
#     sns.histplot(cath_lab_times, bins=30, kde=False, stat='density', label='Data')
#     x = np.linspace(min(cath_lab_times), max(cath_lab_times), 100)
#     pdf = lognorm.pdf(x, cath_shape, loc=cath_loc, scale=cath_scale)
#     plt.plot(x, pdf, label='Fitted Lognormal')
#     plt.title('CATH Lab Times Distribution')
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.show()
    
#     # Plot EP Lab times
#     sns.histplot(ep_lab_times, bins=30, kde=False, stat='density', label='Data', color='orange')
#     x = np.linspace(min(ep_lab_times), max(ep_lab_times), 100)
#     pdf = lognorm.pdf(x, ep_shape, loc=ep_loc, scale=ep_scale)
#     plt.plot(x, pdf, label='Fitted Lognormal', color='red')
#     plt.title('EP Lab Times Distribution')
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.show()
    
#     # Plot Recovery times
#     sns.histplot(recovery_times, bins=30, kde=False, stat='density', label='Data', color='green')
#     x = np.linspace(min(recovery_times), max(recovery_times), 100)
#     pdf = lognorm.pdf(x, recovery_shape, loc=recovery_loc, scale=recovery_scale)
#     plt.plot(x, pdf, label='Fitted Lognormal', color='black')
#     plt.title('Recovery Times Distribution')
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.show()

# # Uncomment the following line to plot the distributions
# # plot_fitted_distributions()

# # 2. Adjust the patient process
# def patient(env, name, patient_type, hospital):
#     """Simulates a patient moving through the hospital system."""
#     arrival_time = env.now
#     hospital.stats['arrivals'] += 1

#     # Check if holding bays are open
#     if not hospital.holding_bays_open:
#         # Holding bays are closed; patient cannot be admitted
#         hospital.stats['cancelled'] += 1  # Procedure is cancelled
#         return  # Patient leaves the system

#     # Request a holding bay for preparation
#     with hospital.holding_bays.request(priority=1) as request:
#         result = yield request | env.timeout(60)  # Wait up to 60 minutes

#         if request in result:
#             # Holding bay allocated
#             preparation_time = np.random.exponential(20)
#             yield env.timeout(preparation_time)

#             # Wait until lab unit is available
#             with hospital.lab_units.request() as lab_request:
#                 yield lab_request
#                 # Release holding bay during procedure
#                 hospital.holding_bays.release(request)

#                 # Procedure time based on patient type
#                 procedure_duration = generate_procedure_time(patient_type)
#                 yield env.timeout(procedure_duration)

#                 # Check if holding bays are open for recovery
#                 if hospital.holding_bays_open:
#                     # Request a holding bay for recovery
#                     with hospital.holding_bays.request(priority=0) as recovery_request:
#                         yield recovery_request
#                         # Release lab unit
#                         hospital.lab_units.release(lab_request)
#                         recovery_duration = generate_recovery_time()
#                         yield env.timeout(recovery_duration)
#                         # Release holding bay after recovery
#                         hospital.holding_bays.release(recovery_request)
#                         hospital.stats['processed'] += 1
#                 else:
#                     # Holding bays are closed; transfer patient to hospital for recovery
#                     # Release lab unit
#                     hospital.lab_units.release(lab_request)
#                     # Record that patient was transferred
#                     hospital.stats['transferred'] += 1
#                     hospital.stats['processed'] += 1  # Considered as processed
#         else:
#             # Patient leaves due to waiting too long
#             hospital.stats['reneged'] += 1

# # 3. Define the Hospital Environment
# class Hospital:
#     """Represents the hospital environment."""
#     def __init__(self, env, num_holding_bays, closing_time):
#         self.env = env
#         self.holding_bays = simpy.PriorityResource(env, capacity=num_holding_bays)
#         self.lab_units = simpy.Resource(env, capacity=12)
#         self.closing_time = closing_time
#         self.holding_bays_open = True  # Holding bays are initially open
#         self.stats = {
#             'arrivals': 0,
#             'processed': 0,
#             'reneged': 0,
#             'cancelled': 0,
#             'transferred': 0,  # To track patients transferred to hospital
#             'holding_bay_idle_time': 0,
#             'lab_unit_idle_time': 0,
#         }
#         self.last_event_time = 0
#         env.process(self.close_holding_bays())

#     def close_holding_bays(self):
#         """Closes the holding bays at the specified closing time."""
#         yield self.env.timeout(self.closing_time)
#         self.holding_bays_open = False  # Holding bays are now closed

# # 4. Define the Arrival Process
# def arrival_process(env, patient_type, hospital):
#     """Generates patients arriving at the hospital."""
#     while True:
#         current_time = env.now
#         arrival_rate = get_arrival_rate(patient_type, current_time)
#         if arrival_rate == 0 or current_time >= hospital.closing_time:
#             # No arrivals during this time or after closing time
#             yield env.timeout(1)
#             continue
#         interarrival_time = random.expovariate(arrival_rate)
#         yield env.timeout(interarrival_time)
#         # Check again if the holding bays are open
#         if env.now >= hospital.closing_time:
#             break  # Stop generating patients
#         patient_id = f"{patient_type}_{hospital.stats['arrivals'] + 1}"
#         env.process(patient(env, patient_id, patient_type, hospital))

# # 5. Get Arrival Rate Based on Time
# def get_arrival_rate(patient_type, current_time):
#     """Returns the arrival rate based on the time of day."""
#     for start, end, rate in ARRIVAL_RATES[patient_type]:
#         if start <= current_time < end:
#             return rate
#     return 0

# # 6. Run the Simulation
# def run_simulation(num_holding_bays, closing_time):
#     """Runs the simulation and returns collected statistics."""
#     env = simpy.Environment()
#     hospital = Hospital(env, num_holding_bays, closing_time)
#     env.process(arrival_process(env, 'CATH', hospital))
#     env.process(arrival_process(env, 'EP', hospital))
#     env.run(until=SIM_TIME)
#     return hospital.stats

# # 7. Main Loop for Experimentation
# # Collect results for different scenarios
# results = []
# for num_bays in HOLDING_BAY_COUNTS:
#     for closing in CLOSING_TIMES:
#         stats = run_simulation(num_bays, closing)
#         results.append({
#             'Holding Bays': num_bays,
#             'Closing Time': int(closing / 60),
#             'Total Arrivals': stats['arrivals'],
#             'Processed': stats['processed'],
#             'Reneged': stats['reneged'],
#             'Cancelled': stats['cancelled'],
#             'Transferred': stats.get('transferred', 0),
#             # Add additional metrics as needed
#         })
#         print(f"Simulation with {num_bays} holding bays and closing time at {int(closing / 60)}:00")
#         print(f"Total Arrivals: {stats['arrivals']}")
#         print(f"Processed: {stats['processed']}")
#         print(f"Reneged: {stats['reneged']}")
#         print(f"Cancelled: {stats['cancelled']}")
#         print(f"Transferred: {stats.get('transferred', 0)}")
#         print("-" * 50)

# # 8. Convert results to DataFrame for analysis
# results_df = pd.DataFrame(results)

# # 9. Save results to CSV (optional)
# results_df.to_csv('simulation_results.csv', index=False)

# # 10. Print the results DataFrame
# print(results_df)
