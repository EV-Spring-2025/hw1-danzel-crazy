import torch
from torch_3dgs.point import get_point_clouds


def dict_to_device(data: dict, device: torch.device) -> dict:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}

def visualize_points_plotly(camera, depth, alpha, rgb):
    point = get_point_clouds(
        camera=camera,
        depth=depth,
        alpha=alpha,
        rgb=rgb,
    )

    # # Create a Plotly 3D scatter plot of the point cloud
    # scatter = go.Scatter3d(
    #     x=point_cloud.coords[:, 0],
    #     y=point_cloud.coords[:, 1],
    #     z=point_cloud.coords[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=2,
    #         color=colors,
    #         opacity=0.8
    #     )
    # )
    
    # layout = go.Layout(
    #     scene=dict(
    #         xaxis_title='X',
    #         yaxis_title='Y',
    #         zaxis_title='Z'
    #     ),
    #     margin=dict(l=0, r=0, b=0, t=0)
    # )
    
    # fig = go.Figure(data=[scatter], layout=layout)
    # fig.show()

