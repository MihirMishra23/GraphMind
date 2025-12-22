import tinker
service_client = tinker.ServiceClient()
# base_model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
base_model = "meta-llama/Llama-3.3-70B-Instruct"
training_client = service_client.create_lora_training_client(
    base_model=base_model
)
sampling_path = training_client.save_weights_for_sampler(name="0000").result().path
print(sampling_path)
