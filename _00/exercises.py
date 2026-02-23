import torch

def execute():
	# 2. create a random tensor with shape (7, 7)
	tensor = torch.rand(7, 7)
	print("Random tensor with shape (7, 7):")
	print(tensor) 

	# 3. create a random tensor with shape (1, 7)
	tensor2 = torch.rand(1, 7)
	print("Random tensor with shape (1, 7):")
	print(tensor2)

	# 3. perform a matrix multiplication between the two tensors (the output should have shape (7, 1))
	tensor3 = torch.matmul(tensor, tensor2.T)
	print("Result of matrix multiplication (shape should be (7, 1)):")
	print(tensor3)

	# 4. repeat 2 and 3 with seed set to 0
	torch.manual_seed(0)
	
	tensor = torch.rand(7, 7)
	print("Random tensor with shape (7, 7) and seed 0:")
	print(tensor)

	tensor2 = torch.rand(1, 7)
	print("Random tensor with shape (1, 7) and seed 0:")
	print(tensor2)

	tensor3 = torch.matmul(tensor, tensor2.T)
	print("Result of matrix multiplication with seed 0 (shape should be (7, 1)):")
	print(tensor3)

	# 8. Find the max and min of tensor3
	max_value = torch.max(tensor3)
	min_value = torch.min(tensor3)
	print(f"Max value in tensor3: {max_value}")
	print(f"Min value in tensor3: {min_value}")

	# 9. find the the max and min indexes of tensor3
	max_index = torch.argmax(tensor3)
	min_index = torch.argmin(tensor3)
	print(f"Max index in tensor3: {max_index}")
	print(f"Min index in tensor3: {min_index}")

	# 10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.
	torch.manual_seed(7)
	tensor = torch.rand(1, 1, 1, 10)
	print("Random tensor with shape (1, 1, 1, 10) and seed 7")
	print(tensor)
	tensor_squeezed = torch.squeeze(tensor)
	print("Squeezed tensor")
	print(tensor_squeezed)