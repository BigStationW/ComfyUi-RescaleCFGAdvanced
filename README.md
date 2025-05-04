# ComfyUi-RescaleCFGAdvanced

## 1) Intro
ReforgeCFG is a ComfyUI node designed to add details to your image. [While it already exists in Comfy Core](https://github.com/comfyanonymous/ComfyUI/blob/80a44b97f5cbcb890896e2b9e65d177f1ac6a588/comfy_extras/nodes_model_advanced.py#L258), it lacks timesteps for adjustment. 

The issue with this is that ReforgeCFG is not intended to be applied throughout the entire process; doing so can result in strange glitches.
![combined_image](https://github.com/user-attachments/assets/b0e0ca59-8aa5-4ea4-83eb-b68d59f9c97e)
As you can see, it works better when you restrict the timesteps range to something like [0.05 -> 0.5].

The “RescaleCFGAdvanced” node is meant to give you this set of choices.

## 2) Install
Navigate to the **ComfyUI/custom_nodes** folder and run the following command in your terminal:

```git clone https://github.com/BigStationW/ComfyUi-RescaleCFGAdvanced```

## 2) Usage
Double click on the empty space of ComfyUi's Node interface and write "RescaleCFGAdvanced"
![image](https://github.com/user-attachments/assets/420a5ce8-b6f8-480f-94a9-754af4a46c9e)
