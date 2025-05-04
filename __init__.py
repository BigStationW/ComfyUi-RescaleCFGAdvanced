import torch
import math
import logging

class RescaleCFGAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "multiplier": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model" 

    def patch(self, model, multiplier, start_percent, end_percent):
        #logging.info(f"[RescaleCFGAdvanced Patch] Initial inputs: multiplier={multiplier}, start={start_percent}, end={end_percent}")

        # Ensure start is less than or equal to end for logical consistency
        if start_percent > end_percent:
            start_percent, end_percent = end_percent, start_percent
            #logging.info(f"[RescaleCFGAdvanced Patch] Swapped start/end percentages: New start={start_percent}, New end={end_percent}")

        effective_start_percent = start_percent
        effective_end_percent = end_percent

        def rescale_cfg_advanced_wrapper(args):
            nonlocal effective_start_percent, effective_end_percent, multiplier

            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            current_sigma_tensor = args["sigma"] # Shape (batch_size,)
            x_orig = args["input"]

            # Basic validation
            if cond is None or uncond is None or cond_scale is None or current_sigma_tensor is None or current_sigma_tensor.numel() == 0 or x_orig is None:
                 #logging.error(f"[RescaleCFGAdvanced Inner] Missing critical args. Cannot proceed.")
                 denoised = args.get("uncond_denoised", args.get("cond_denoised", None))
                 if denoised is not None:
                      #logging.warning("[RescaleCFGAdvanced Inner] Returning raw model output due to missing args.")
                      return denoised
                 else: 
                      raise ValueError("[RescaleCFGAdvanced Inner] Cannot proceed due to missing critical arguments.")


            current_sigma = current_sigma_tensor[0].item() 
            log_prefix = f"[RescaleCFGAdvanced Inner Sigma={current_sigma:.4f}]"

            apply_rescale = False

            sample_sigmas = None
            model_options = args.get("model_options", {})
            if isinstance(model_options, dict):
                transformer_options = model_options.get("transformer_options", {})
                if isinstance(transformer_options, dict):
                    sample_sigmas = transformer_options.get("sample_sigmas", None)

            if sample_sigmas is not None and isinstance(sample_sigmas, torch.Tensor) and sample_sigmas.numel() > 1:
                total_steps_in_schedule = len(sample_sigmas) 
                schedule_indices = total_steps_in_schedule - 1

                if schedule_indices <= 0:
                    #logging.warning(f"{log_prefix} Sigma schedule has invalid length {total_steps_in_schedule}. Cannot calculate percentage.")
                    current_step_index = 0
                else:
                    found_index = -1
                    matches = (torch.isclose(current_sigma_tensor[0], sample_sigmas, atol=1e-5, rtol=1e-5)).nonzero()
                    if len(matches) > 0:
                        found_index = matches[0].item()
                    else:
                        for i in range(schedule_indices):
                            s_curr = sample_sigmas[i].item()
                            s_next = sample_sigmas[i+1].item()

                            if s_curr >= current_sigma > s_next:
                                 found_index = i 
                                 break
                            
                        if found_index == -1:
                            if current_sigma >= sample_sigmas[0].item():
                                found_index = 0
                            elif current_sigma <= sample_sigmas[-1].item():
                                found_index = schedule_indices

                    if found_index == -1:
                        #logging.warning(f"{log_prefix} Could not reliably find current sigma index in schedule {sample_sigmas}. Defaulting index to 0.")
                        current_step_index = 0
                    else:
                        current_step_index = found_index

                current_percent = current_step_index / schedule_indices if schedule_indices > 0 else 0.0

                #logging.info(f"{log_prefix} Using sample_sigmas ({total_steps_in_schedule} sigmas). Current index={current_step_index}, Percent={current_percent:.3f}. Target range=[{effective_start_percent:.3f}, {effective_end_percent:.3f}]")

                tolerance = 1e-5
                if (current_percent >= effective_start_percent - tolerance) and \
                   (current_percent <= effective_end_percent + tolerance):
                    apply_rescale = True

                if effective_start_percent == 0.0 and current_step_index == 0:
                    apply_rescale = True
                if effective_end_percent == 1.0 and current_step_index == schedule_indices:
                     apply_rescale = True

            else:
                #logging.warning(f"{log_prefix} 'sample_sigmas' not found in args['model_options']['transformer_options']. Cannot use step percentage. RescaleCFG range disabled unless range is [0, 1].")
                if effective_start_percent <= 0.0 and effective_end_percent >= 1.0:
                    apply_rescale = True

            if apply_rescale:
                #logging.info(f"{log_prefix} Applying rescale (multiplier: {multiplier}).")
                sigma_view = current_sigma_tensor.view(current_sigma_tensor.shape[:1] + (1,) * (cond.ndim - 1))
                x = x_orig / (sigma_view * sigma_view + 1.0)
                pred_x0_cond = x_orig - cond
                pred_x0_uncond = x_orig - uncond
                epsilon_sigma = 1e-9
                v_pred_cond = ((x - pred_x0_cond) * (sigma_view ** 2 + 1.0) ** 0.5) / (sigma_view + epsilon_sigma)
                v_pred_uncond = ((x - pred_x0_uncond) * (sigma_view ** 2 + 1.0) ** 0.5) / (sigma_view + epsilon_sigma)
                v_pred_cfg = v_pred_uncond + cond_scale * (v_pred_cond - v_pred_uncond)
                ro_pos = torch.std(v_pred_cond, dim=tuple(range(1, v_pred_cond.ndim)), keepdim=True)
                ro_cfg = torch.std(v_pred_cfg, dim=tuple(range(1, v_pred_cfg.ndim)), keepdim=True)
                epsilon_std = 1e-5
                factor = ro_pos / (ro_cfg + epsilon_std)
                factor = torch.nan_to_num(factor, nan=1.0, posinf=1.0, neginf=1.0)
                v_pred_rescaled = v_pred_cfg * factor
                v_pred_final = multiplier * v_pred_rescaled + (1.0 - multiplier) * v_pred_cfg
                pred_x0_final = x - v_pred_final * sigma_view / (sigma_view * sigma_view + 1.0) ** 0.5
                final_output = x_orig - pred_x0_final

                return final_output
            else:
                standard_cfg_output = uncond + cond_scale * (cond - uncond)
                return standard_cfg_output

        # --- Apply the patch ---
        m = model.clone()
        m.set_model_sampler_cfg_function(rescale_cfg_advanced_wrapper)
        #logging.info("[RescaleCFGAdvanced Patch] Patch applied successfully using sample_sigmas percentage.")
        return (m, )

NODE_CLASS_MAPPINGS = {
    "RescaleCFGAdvanced": RescaleCFGAdvanced, 
}