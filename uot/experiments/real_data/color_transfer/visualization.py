import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
from PIL import Image
import numpy as np
import argparse

class OptimalTransportGallery:
    def __init__(self, origin_folder, results_folder):
        self.origin_folder = origin_folder
        self.results_folder = results_folder
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def get_origin_images(self):
        """Get all image files from the origin folder"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        images = []
        for file in os.listdir(self.origin_folder):
            if os.path.splitext(file.lower())[1] in valid_extensions:
                images.append(file)
        return sorted(images)
    
    def get_result_images(self, source_img, target_img):
        """Get all result images that match source and target pattern"""
        if not source_img or not target_img:
            return []

        pattern = f"{source_img}_{target_img}_"
        
        results = []
        for file in os.listdir(self.results_folder):
            if file.startswith(pattern):
                results.append(file)
        return sorted(results)
    
    def load_image(self, folder, filename, target_size=(400, 300)):
        """Load image with center crop to maintain aspect ratio"""
        if not filename:
            return None
        
        try:
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            target_ratio = target_size[0] / target_size[1]
            img_ratio = img.width / img.height
            
            if img_ratio > target_ratio:
                new_width = int(img.height * target_ratio)
                left = (img.width - new_width) // 2
                crop_box = (left, 0, left + new_width, img.height)
            else:
                new_height = int(img.width / target_ratio)
                top = (img.height - new_height) // 2
                crop_box = (0, top, img.width, top + new_height)
            
            img = img.crop(crop_box)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            return np.flipud(np.array(img))
        except Exception as e:
            raise ValueError(f"Error loading image {filename}: {e}")

    def setup_layout(self):
        """Setup the dashboard layout"""
        origin_images = self.get_origin_images()
        
        self.app.layout = html.Div([
            html.H1("OT Color Transfer Gallery Dashboard", 
                style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.Div([
                html.Div([
                    html.Label("Source Image:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='source-dropdown',
                        options=[{'label': img, 'value': img} for img in origin_images],
                        value=origin_images[0] if origin_images else None,
                        style={'width': '100%'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                
                html.Div([
                    html.Label("Result Image:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='result-dropdown',
                        options=[],
                        value=None,
                        style={'width': '100%'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                
                html.Div([
                    html.Label("Target Image:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='target-dropdown',
                        options=[{'label': img, 'value': img} for img in origin_images],
                        value=origin_images[1] if len(origin_images) > 1 else None,
                        style={'width': '100%'}
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ], style={'marginBottom': '30px', 'padding': '20px'}),
            
            html.Div([
                dcc.Graph(id='image-gallery', style={'height': '600px'})
            ])
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('result-dropdown', 'options'),
            Output('result-dropdown', 'value'),
            [Input('source-dropdown', 'value'),
             Input('target-dropdown', 'value')]
        )
        def update_result_dropdown(source_img, target_img):
            """Update result dropdown based on source and target selection"""
            result_images = self.get_result_images(source_img, target_img)
            options = [{'label': img, 'value': img} for img in result_images]
            value = result_images[0] if result_images else None
            return options, value
        
        @self.app.callback(
            Output('image-gallery', 'figure'),
            [Input('source-dropdown', 'value'),
             Input('target-dropdown', 'value'),
             Input('result-dropdown', 'value')]
        )
        def update_gallery(source_img, target_img, result_img):
            """Update the three-panel image gallery"""
            

            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Source Image', 'Result Image', 'Target Image'),
                specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
            )

            source_array = self.load_image(self.origin_folder, source_img)
            if source_array is not None:
                fig.add_trace(
                    go.Image(z=source_array),
                    row=1, col=1
                )
            
            result_array = self.load_image(self.results_folder, result_img)
            if result_array is not None:
                fig.add_trace(
                    go.Image(z=result_array),
                    row=1, col=2
                )
            
            target_array = self.load_image(self.origin_folder, target_img)
            if target_array is not None:
                fig.add_trace(
                    go.Image(z=target_array),
                    row=1, col=3
                )
            
            fig.update_layout(
                showlegend=False,
                height=600,
                margin=dict(l=20, r=20, t=20, b=20)
            )

            for i in range(1, 4):
                fig.update_xaxes(
                    showticklabels=False, 
                    showgrid=False, 
                    zeroline=False,
                    range=[0, 400],
                    row=1, col=i
                )
                fig.update_yaxes(
                    showticklabels=False, 
                    showgrid=False, 
                    zeroline=False,
                    range=[0, 300],
                    scaleanchor=f"x{i}",
                    scaleratio=1,
                    row=1, col=i
                )
            
            return fig
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        self.app.run(debug=debug, port=port)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the Optimal Transport Gallery Dashboard")

    parser.add_argument(
        "--origin_folder",
        type=str,
        required=True,
        help="Path to the folder containing the original images"
    )

    parser.add_argument(
        "--results_folder",
        type=str,
        required=True,
        help="Path to the folder containing the result images"
    )

    args = parser.parse_args()

    gallery = OptimalTransportGallery(args.origin_folder, args.results_folder)
    gallery.run_server(debug=False, port=8050)
