import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LoadData:
    def __init__(self):
        # Infer Data
        self.targets = np.load("gat_artifacts/targets.npy")
        self.targets_global_anom = np.load("gat_artifacts/targets_global_anom.npy")
        self.targets_trend_anom = np.load("gat_artifacts/targets_trend_anom.npy")
        # Predictions
        self.preds = np.load("gat_artifacts/preds.npy")
        self.preds_global_anom = np.load("gat_artifacts/preds_global_anom.npy")
        self.preds_trend_anom = np.load("gat_artifacts/preds_trend_anom.npy")
        # Spatial attention
        self.attn_f = np.load("gat_artifacts/attn_f.npy")
        self.attn_f_global_anom = np.load("gat_artifacts/attn_f_global_anom.npy")
        self.attn_f_trend_anom = np.load("gat_artifacts/attn_f_trend_anom.npy")
        # Temporal attention
        self.attn_t = np.load("gat_artifacts/attn_t.npy")
        self.attn_t_global_anom = np.load("gat_artifacts/attn_t_global_anom.npy")
        self.attn_t_trend_anom = np.load("gat_artifacts/attn_t_trend_anom.npy")
        # Edge idx
        self.attn_f_edge_idx = np.load("gat_artifacts/attn_f_edge_idx.npy")
        self.attn_t_edge_idx = np.load("gat_artifacts/attn_t_edge_idx.npy")
        # Metadata
        self.num_sensors = self.targets.shape[1]
        self.time_steps = self.targets.shape[0]
        self.num_attn_heads = self.attn_f.shape[2]

class Viz:
    def __init__(self, data):
        self.data = data
    
    def perform_nominal(self, attn_head=0,dt=10):
        # Number rows
        num_rows = self.data.num_sensors
        # Create figure
        fig = make_subplots(rows=num_rows, cols=3)

        # Line Plots
        num_line_plots = 0
        for sensor in range(self.data.num_sensors):
            # Line plots
            fig.add_trace(go.Scatter(y=self.data.targets[:,sensor], mode="lines", line=dict(color='teal', width=2)), row=sensor+1, col=1)
            fig.add_trace(go.Scatter(y=self.data.preds[:,sensor], mode="lines", line=dict(color='purple', width=2)), row=sensor+1, col=1)
            num_line_plots += 2

            # Spatial attention background line plots
            for step in range(0,self.data.time_steps,dt):
                attn_f_sensor_idx = self.data.attn_f_edge_idx.T[:,1]==sensor
                attn_f_sensor = self.data.attn_f[step,attn_f_sensor_idx,attn_head]
                fig.add_trace(go.Scatter(y=attn_f_sensor, mode="lines", line=dict(color='grey', width=2)), row=sensor+1, col=2)
                num_line_plots += 1

            # Temporal attention background line plots
            for step in range(0,self.data.time_steps,dt):
                attn_t_sensor_idx = self.data.attn_t_edge_idx.T[:,1]==sensor
                attn_t_sensor = self.data.attn_t[step,attn_t_sensor_idx,attn_head]
                fig.add_trace(go.Scatter(y=attn_t_sensor, mode="lines", line=dict(color='grey', width=2)), row=sensor+1, col=3)
                num_line_plots += 1

        # Dot Plots (time step indicator on line plot with slider)
        num_dots = 0
        for sensor in range(self.data.num_sensors):
            for step in range(0,self.data.time_steps,dt):
                fig.add_trace(go.Scatter(x=[step], y=[self.data.preds[step,sensor]], mode="markers", marker=dict(color='red',size=10,)), row=sensor+1, col=1)
                num_dots += 1

        # Spatial Attention Visualization
        num_attn_f_plots = 0
        for sensor in range(self.data.num_sensors):
            """
            0. Select attention based on sensor and attention head
            1. Create graph centered around the ith sensor
                - insert nodes
                - insert edges
                - define node size based on attention
            2. Insert graph in plotly
            """
            for step in range(0,self.data.time_steps,dt):
                attn_f_sensor_idx = self.data.attn_f_edge_idx.T[:,1]==sensor
                attn_f_sensor = self.data.attn_f[step,attn_f_sensor_idx,attn_head]
                fig.add_trace(go.Scatter(y=attn_f_sensor, mode="lines", line=dict(color='red', width=2)), row=sensor+1, col=2)
                num_attn_f_plots += 1

        # Temporal Attention Visualization
        num_attn_t_plots = 0
        for sensor in range(self.data.num_sensors):
            """
            0. Select attention based on sensor and attention head
            1. Create graph centered around the ith sensor
                - insert nodes
                - insert edges
                - define node size based on attention
            2. Insert graph in plotly
            """
            for step in range(0,self.data.time_steps,dt):
                attn_t_sensor_idx = self.data.attn_t_edge_idx.T[:,1]==sensor
                attn_t_sensor = self.data.attn_t[step,attn_t_sensor_idx,attn_head]
                fig.add_trace(go.Scatter(y=attn_t_sensor, mode="lines", line=dict(color='red', width=2)), row=sensor+1, col=3)
                num_attn_t_plots += 1

        # Slider
        # source: https://community.plotly.com/t/using-one-slider-to-control-multiple-subplots-not-multiple-traces/13955/4 
        steps = []
        for i,idx in enumerate(range(0,self.data.time_steps,dt)):
            num_time_steps = len(range(0,self.data.time_steps,dt))
            step = dict(
                method = 'restyle',  
                args = ['visible', ['legendonly'] * len(fig.data)],
            )
            # Set Col 1 Line Plots Visibility
            for col_1_plot in range(num_line_plots):
                step['args'][1][col_1_plot] = True

            # Set Col 1 Dots Visibility
            for sensor in range(self.data.num_sensors):
                plot_num = num_line_plots + (num_time_steps)*sensor + i
                step['args'][1][plot_num] = True

            # Set Col 2 Spatial Attention Plots Visibility
            for sensor in range(self.data.num_sensors):
                plot_num = num_line_plots + num_dots + (num_time_steps)*sensor + i
                step['args'][1][plot_num] = True

            # Set Col 3 Temporal Attention Plots Visibility
            for sensor in range(self.data.num_sensors):
                plot_num = num_line_plots + num_dots + num_attn_f_plots + (num_time_steps)*sensor + i
                step['args'][1][plot_num] = True

            steps.append(step)

        sliders = [dict(steps = steps,)]

        fig.layout.sliders = sliders
        fig.update_layout(height=400*num_rows, width=1500,title_text="NOMINAL - Inference Results")

        fig.show()        
        fig.write_html(f"gat_artifacts/nominal.html")

    def perform_anom(self, attn_head=0,dt=10,anom_type="GLOBAL"):
        # Number rows
        num_rows = self.data.num_sensors
        # Create figure
        fig = make_subplots(rows=num_rows, cols=3)

        '''
        targets_anom
        preds_anom
        attn_f_anom
        attn_f_anom_edge_idx
        attn_t_anom_edge_idx
        attn_t_anom
        '''
        targets_anom = self.data.targets_global_anom if anom_type=="GLOBAL" else self.data.targets_trend_anom
        preds_anom  = self.data.preds_global_anom if anom_type=="GLOBAL" else self.data.preds_trend_anom
        attn_f_anom  = self.data.attn_f_global_anom if anom_type=="GLOBAL" else self.data.attn_f_trend_anom
        attn_t_anom  = self.data.attn_t_global_anom if anom_type=="GLOBAL" else self.data.attn_t_trend_anom
        attn_f_anom_edge_idx  = self.data.attn_f_edge_idx if anom_type=="GLOBAL" else self.data.attn_f_edge_idx
        attn_t_anom_edge_idx  = self.data.attn_t_edge_idx if anom_type=="GLOBAL" else self.data.attn_t_edge_idx

        # Line Plots
        num_line_plots = 0
        for sensor in range(self.data.num_sensors):
            # Line plots
            fig.add_trace(go.Scatter(y=targets_anom[:,sensor], mode="lines", line=dict(color='red', width=2)), row=sensor+1, col=1)
            fig.add_trace(go.Scatter(y=self.data.targets[:,sensor], mode="lines", line=dict(color='teal', width=2)), row=sensor+1, col=1)
            fig.add_trace(go.Scatter(y=preds_anom[:,sensor], mode="lines", line=dict(color='purple', width=2)), row=sensor+1, col=1)
            num_line_plots += 3

            # Spatial attention background line plots
            for step in range(0,self.data.time_steps,dt):
                attn_f_sensor_idx = attn_f_anom_edge_idx.T[:,1]==sensor
                attn_f_sensor = attn_f_anom[step,attn_f_sensor_idx,attn_head]
                fig.add_trace(go.Scatter(y=attn_f_sensor, mode="lines", line=dict(color='grey', width=2)), row=sensor+1, col=2)
                num_line_plots += 1

            # Temporal attention background line plots
            for step in range(0,self.data.time_steps,dt):
                attn_t_sensor_idx = attn_t_anom_edge_idx.T[:,1]==sensor
                attn_t_sensor = attn_t_anom[step,attn_t_sensor_idx,attn_head]
                fig.add_trace(go.Scatter(y=attn_t_sensor, mode="lines", line=dict(color='grey', width=2)), row=sensor+1, col=3)
                num_line_plots += 1

        # Dot Plots (time step indicator on line plot with slider)
        num_dots = 0
        for sensor in range(self.data.num_sensors):
            for step in range(0,self.data.time_steps,dt):
                fig.add_trace(go.Scatter(x=[step], y=[preds_anom[step,sensor]], mode="markers", marker=dict(color='red',size=10,)), row=sensor+1, col=1)
                num_dots += 1

        # Spatial Attention Visualization
        num_attn_f_plots = 0
        for sensor in range(self.data.num_sensors):
            """
            0. Select attention based on sensor and attention head
            1. Create graph centered around the ith sensor
                - insert nodes
                - insert edges
                - define node size based on attention
            2. Insert graph in plotly
            """
            for step in range(0,self.data.time_steps,dt):
                attn_f_sensor_idx = attn_f_anom_edge_idx.T[:,1]==sensor
                attn_f_sensor = attn_f_anom[step,attn_f_sensor_idx,attn_head]
                fig.add_trace(go.Scatter(y=attn_f_sensor, mode="lines", line=dict(color='red', width=2)), row=sensor+1, col=2)
                num_attn_f_plots += 1

        # Temporal Attention Visualization
        num_attn_t_plots = 0
        for sensor in range(self.data.num_sensors):
            """
            0. Select attention based on sensor and attention head
            1. Create graph centered around the ith sensor
                - insert nodes
                - insert edges
                - define node size based on attention
            2. Insert graph in plotly
            """
            for step in range(0,self.data.time_steps,dt):
                attn_t_sensor_idx = attn_t_anom_edge_idx.T[:,1]==sensor
                attn_t_sensor = attn_t_anom[step,attn_t_sensor_idx,attn_head]
                fig.add_trace(go.Scatter(y=attn_t_sensor, mode="lines", line=dict(color='red', width=2)), row=sensor+1, col=3)
                num_attn_t_plots += 1

        # Slider
        # source: https://community.plotly.com/t/using-one-slider-to-control-multiple-subplots-not-multiple-traces/13955/4 
        steps = []
        for i,idx in enumerate(range(0,self.data.time_steps,dt)):
            num_time_steps = len(range(0,self.data.time_steps,dt))
            step = dict(
                method = 'restyle',  
                args = ['visible', ['legendonly'] * len(fig.data)],
            )
            # Set Col 1 Line Plots Visibility
            for col_1_plot in range(num_line_plots):
                step['args'][1][col_1_plot] = True

            # Set Col 1 Dots Visibility
            for sensor in range(self.data.num_sensors):
                plot_num = num_line_plots + (num_time_steps)*sensor + i
                step['args'][1][plot_num] = True

            # Set Col 2 Spatial Attention Plots Visibility
            for sensor in range(self.data.num_sensors):
                plot_num = num_line_plots + num_dots + (num_time_steps)*sensor + i
                step['args'][1][plot_num] = True

            # Set Col 3 Temporal Attention Plots Visibility
            for sensor in range(self.data.num_sensors):
                plot_num = num_line_plots + num_dots + num_attn_f_plots + (num_time_steps)*sensor + i
                step['args'][1][plot_num] = True

            steps.append(step)

        sliders = [dict(steps = steps,)]

        fig.layout.sliders = sliders
        fig.update_layout(height=400*num_rows, width=1500,title_text=f"{anom_type} ANOMALY - Inference Results")

        fig.show()        
        fig.write_html(f"gat_artifacts/{anom_type}.html")


if __name__ == "__main__":
    data = LoadData()
    viz = Viz(data)
    viz.perform_nominal(attn_head=0,dt=20)
    viz.perform_anom(attn_head=0,dt=20,anom_type="TREND")
    
