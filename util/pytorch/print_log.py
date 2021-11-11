def print_model_info(model, optimizer, loss_func, verbose=1):
    # Print model's state_dict
    if verbose > 0:
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    if verbose > 1:
        print(f"\nLoss function: {str(loss_func)}")

    if verbose > 2:
        print()
        # Print optimizer's state_dict
        print("\nOptimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
