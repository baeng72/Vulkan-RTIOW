#include "Vulkan.h"

const int WIDTH = 800;
const int HEIGHT = 600;



int main() {
	GLFWwindow* window = initWindow("RTIOW: Ch.05 - Sphere",WIDTH, HEIGHT);


	uint32_t extCount = 0;
	auto ext = glfwGetRequiredInstanceExtensions(&extCount);
	std::vector<const char*> requiredExtensions(ext, ext + extCount);
#ifdef NDEBUG
	std::vector<const char*> requiredLayers{ "VK_LAYER_LUNARG_monitor" };
#else
	std::vector<const char*> requiredLayers{ "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor" };
#endif

	VkInstance instance = initInstance(requiredExtensions, requiredLayers);
	VkSurfaceKHR surface = initSurface(instance, window);

	Queues queues;
	VkPhysicalDevice physicalDevice = choosePhysicalDevice(instance, surface, queues);
	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
	VkSurfaceCapabilitiesKHR surfaceCaps;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps);
	uint32_t formatCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
	std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, surfaceFormats.data());
	uint32_t presentModeCount = 0;
	vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
	std::vector<VkPresentModeKHR> presentModes(presentModeCount);
	vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());


	std::vector<const char*> deviceExtensions{ "VK_KHR_swapchain" };
	VkPhysicalDeviceFeatures enabledFeatures{};
	VkDevice device = initDevice(physicalDevice, deviceExtensions, queues, enabledFeatures);

	VkQueue graphicsQueue = getDeviceQueue(device, queues.graphicsQueueFamily);
	VkQueue presentQueue = getDeviceQueue(device, queues.presentQueueFamily);
	VkQueue computeQueue = getDeviceQueue(device, queues.computeQueueFamily);

	VkPresentModeKHR presentMode = chooseSwapchainPresentMode(presentModes);
	VkSurfaceFormatKHR swapchainFormat = chooseSwapchainFormat(surfaceFormats);
	VkFormatProperties formatProperties;
	vkGetPhysicalDeviceFormatProperties(physicalDevice, swapchainFormat.format, &formatProperties);
	VkExtent2D swapchainExtent;
	swapchainExtent.width = WIDTH;
	swapchainExtent.height = HEIGHT;
	uint32_t numImages = 2;
	VkSwapchainKHR swapchain = initSwapchain(device, surface, surfaceCaps, presentMode, swapchainFormat, swapchainExtent,numImages);
	std::vector<VkImage> swapchainImages;
	getSwapchainImages(device, swapchain, swapchainImages);
	std::vector<VkImageView> swapchainImageViews;
	initSwapchainImageViews(device, swapchainImages, swapchainFormat.format, swapchainImageViews);

	VkSemaphore presentComplete = initSemaphore(device);
	VkSemaphore renderComplete = initSemaphore(device);

	VkCommandPool commandPool = initCommandPool(device, queues.graphicsQueueFamily);
	VkCommandBuffer commandBuffer = initCommandBuffer(device, commandPool);

	VkCommandPool computeCommandPool = initCommandPool(device, queues.computeQueueFamily);
	std::vector<VkCommandBuffer> computeCommandBuffers(swapchainImages.size());
	for (size_t i = 0; i < swapchainImages.size(); i++) {
		VkCommandBuffer computeCommandBuffer = initCommandBuffer(device, computeCommandPool);
		computeCommandBuffers[i] = computeCommandBuffer;
	}
	std::vector<VkFence> computeFences(swapchainImages.size());
	for (size_t i = 0; i < swapchainImages.size(); i++) {
		VkFence computeFence = initFence(device, VK_FENCE_CREATE_SIGNALED_BIT);
		computeFences[i] = computeFence;
	}

	Image computeImage;
#define COMPUTE_WIDTH 800
#define COMPUTE_HEIGHT 608
	float aspectRatio =  (float)COMPUTE_WIDTH / (float)COMPUTE_HEIGHT;
	float imageWidth = (float)COMPUTE_WIDTH;// 1024;
	float imageHeight = (float)COMPUTE_WIDTH / aspectRatio;
	initImage(device, swapchainFormat.format, formatProperties, memoryProperties, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, (uint32_t)imageWidth, (uint32_t)imageHeight, computeImage);
	transitionImage(device, graphicsQueue, commandBuffer, computeImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

	struct {
		float imageWidth;
		float imageHeight;
		float viewportWidth;
		float viewportHeight;
		float focalLength;
	}ubo;
	ubo.imageWidth = imageWidth;
	ubo.imageHeight = imageHeight;
	ubo.viewportHeight = 2.0f/aspectRatio;
	ubo.viewportWidth = 2.0f;
	ubo.focalLength = 1.0f;
	Buffer uboBuffer;
	initBuffer(device, memoryProperties, sizeof(ubo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboBuffer);
	void* pUbo = mapBuffer(device, uboBuffer);
	memcpy(pUbo, &ubo, sizeof(ubo));


	std::vector<VkDescriptorSetLayoutBinding> computeLayoutSetBindings = {
		{0,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
		{1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}

	};
	VkDescriptorSetLayout computeDescriptorSetLayout = initDescriptorSetLayout(device, computeLayoutSetBindings);
	std::vector<VkDescriptorPoolSize> computePoolSizes = {
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,3 },
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,3}
	};
	VkDescriptorPool computeDescriptorPool = initDescriptorPool(device, computePoolSizes, 4);

	VkDescriptorSet computeDescriptorSet = initDescriptorSet(device, computeDescriptorSetLayout, computeDescriptorPool);
	VkDescriptorImageInfo imageInfo{};
	imageInfo.sampler = computeImage.sampler;
	imageInfo.imageView = computeImage.imageView;
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	VkDescriptorBufferInfo bufferInfo{};
	bufferInfo.buffer = uboBuffer.buffer;
	bufferInfo.range = sizeof(ubo);
	std::vector<VkWriteDescriptorSet> computeDescriptorWrites = {
		{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,computeDescriptorSet,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,&imageInfo,nullptr,nullptr},
		{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,computeDescriptorSet,1,0,1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,nullptr,&bufferInfo,nullptr}
	};
	updateDescriptorSets(device, computeDescriptorWrites);

	VkPipelineLayout computePipelineLayout = initPipelineLayout(device, computeDescriptorSetLayout);
	VkShaderModule compShader = initShaderModule(device, "Shaders/raytrace05.comp.spv");
	ShaderModule shader = { compShader,VK_SHADER_STAGE_COMPUTE_BIT };
	VkPipeline computePipeline = initComputePipeline(device, computePipelineLayout, shader);
	cleanupShaderModule(device, compShader);
	
	std::vector<VkCommandBuffer> commandBuffers;
	initCommandBuffers(device, commandPool,(uint32_t)swapchainImages.size(), commandBuffers);
	std::vector<VkFence> renderFences(swapchainImages.size());
	for (size_t i = 0; i < swapchainImages.size(); i++) {
		VkFence fence = initFence(device, VK_FENCE_CREATE_SIGNALED_BIT);
		renderFences[i] = fence;
	}

	VkRenderPass renderPass = initRenderPass(device, swapchainFormat.format);

	std::vector<VkFramebuffer> framebuffers;
	initFramebuffers(device, renderPass, swapchainImageViews, WIDTH, HEIGHT, framebuffers);


	float vertices[] = {
		-1.0f,-1.0f,0.0f,0.0f,0.0f,//top left
		1.0f,-1.0f,0.0f,1.0f,0.0f,//top right
		1.0f,1.0f,0.0f,1.0f,1.0f,//bottom right
		-1.0f,1.0f,0.0f,0.0f,1.0f//bottom left;
	};
	
	uint32_t indices[] = {
		0,1,2,0,2,3
	};
	Buffer vertexBuffer;
	Buffer indexBuffer;
	VkDeviceSize maxSize = max(sizeof(vertices), sizeof(indices));
	Buffer stagingBuffer;
	initBuffer(device, memoryProperties, maxSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer);
	void* ptr = mapBuffer(device, stagingBuffer);
	initBuffer(device, memoryProperties, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer);
	initBuffer(device, memoryProperties, sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer);
	//upload vertices
	memcpy(ptr, vertices, sizeof(vertices));
	CopyBufferTo(device, graphicsQueue, commandBuffer, stagingBuffer, vertexBuffer, sizeof(vertices));
	memcpy(ptr, indices, sizeof(indices));
	CopyBufferTo(device, graphicsQueue, commandBuffer, stagingBuffer, indexBuffer, sizeof(indices));


	unmapBuffer(device, stagingBuffer);
	cleanupBuffer(device, stagingBuffer);

	VkShaderModule vertShader = initShaderModule(device, "Shaders/rt.vert.spv");
	VkShaderModule fragShader = initShaderModule(device, "Shaders/rt.frag.spv");
	std::vector<ShaderModule> shaders = { {vertShader,VK_SHADER_STAGE_VERTEX_BIT},{fragShader,VK_SHADER_STAGE_FRAGMENT_BIT} };

	VkVertexInputBindingDescription bindingDescription = { 0,sizeof(float) * 5,VK_VERTEX_INPUT_RATE_VERTEX };
	std::vector<VkVertexInputAttributeDescription> attributeDescriptions = {
		{0,0,VK_FORMAT_R32G32B32_SFLOAT,0},
		{1,0,VK_FORMAT_R32G32_SFLOAT,3 * sizeof(float)}

	};

	std::vector<VkDescriptorSetLayoutBinding> layoutSetBindings = {
		{0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},

	};
	VkDescriptorSetLayout descriptorSetLayout = initDescriptorSetLayout(device, layoutSetBindings);
	std::vector<VkDescriptorPoolSize> poolSizes = {
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,3 }
	};
	VkDescriptorPool descriptorPool = initDescriptorPool(device, poolSizes, 4);

	VkDescriptorSet descriptorSet = initDescriptorSet(device, descriptorSetLayout, descriptorPool);

	//VkDescriptorImageInfo imageInfo{};
	imageInfo.sampler = computeImage.sampler;
	imageInfo.imageView = computeImage.imageView;
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	std::vector<VkWriteDescriptorSet> descriptorWrites = {
		{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,descriptorSet,0,0,1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,&imageInfo,nullptr,nullptr}
	};
	updateDescriptorSets(device, descriptorWrites);

	////VkDescriptorImageInfo imageInfo{};
	//imageInfo.sampler = computeImage.sampler;
	//imageInfo.imageView = computeImage.imageView;
	//imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	//std::vector<VkWriteDescriptorSet> computeDescriptorWrites = {
	//	{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,computeDescriptorSet,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,&imageInfo,nullptr,nullptr}
	//};

	VkPipelineLayout pipelineLayout = initPipelineLayout(device, descriptorSetLayout);
	VkPipeline pipeline = initGraphicsPipeline(device, renderPass, pipelineLayout, swapchainExtent, shaders, bindingDescription, attributeDescriptions);

	cleanupShaderModule(device, vertShader);
	cleanupShaderModule(device, fragShader);

	PFN_vkAcquireNextImageKHR pvkAcquireNextImage = (PFN_vkAcquireNextImageKHR)vkGetDeviceProcAddr(device, "vkAcquireNextImageKHR");
	assert(pvkAcquireNextImage);
	PFN_vkQueuePresentKHR pvkQueuePresent = (PFN_vkQueuePresentKHR)vkGetDeviceProcAddr(device, "vkQueuePresentKHR");
	assert(pvkQueuePresent);
	PFN_vkQueueSubmit pvkQueueSubmit = (PFN_vkQueueSubmit)vkGetDeviceProcAddr(device, "vkQueueSubmit");
	assert(pvkQueueSubmit);
	PFN_vkBeginCommandBuffer pvkBeginCommandBuffer = (PFN_vkBeginCommandBuffer)vkGetDeviceProcAddr(device, "vkBeginCommandBuffer");
	assert(pvkBeginCommandBuffer);
	PFN_vkCmdBeginRenderPass pvkCmdBeginRenderPass = (PFN_vkCmdBeginRenderPass)vkGetDeviceProcAddr(device, "vkCmdBeginRenderPass");
	assert(pvkCmdBeginRenderPass);
	PFN_vkCmdBindPipeline pvkCmdBindPipeline = (PFN_vkCmdBindPipeline)vkGetDeviceProcAddr(device, "vkCmdBindPipeline");
	assert(pvkCmdBindPipeline);
	PFN_vkCmdBindVertexBuffers pvkCmdBindVertexBuffers = (PFN_vkCmdBindVertexBuffers)vkGetDeviceProcAddr(device, "vkCmdBindVertexBuffers");
	assert(pvkCmdBindVertexBuffers);
	PFN_vkCmdBindIndexBuffer pvkCmdBindIndexBuffer = (PFN_vkCmdBindIndexBuffer)vkGetDeviceProcAddr(device, "vkCmdBindIndexBuffer");
	assert(pvkCmdBindIndexBuffer);
	PFN_vkCmdDrawIndexed pvkCmdDrawIndexed = (PFN_vkCmdDrawIndexed)vkGetDeviceProcAddr(device, "vkCmdDrawIndexed");
	assert(pvkCmdDrawIndexed);
	PFN_vkCmdEndRenderPass pvkCmdEndRenderPass = (PFN_vkCmdEndRenderPass)vkGetDeviceProcAddr(device, "vkCmdEndRenderPass");
	assert(pvkCmdEndRenderPass);
	PFN_vkEndCommandBuffer pvkEndCommandBuffer = (PFN_vkEndCommandBuffer)vkGetDeviceProcAddr(device, "vkEndCommandBuffer");
	assert(pvkEndCommandBuffer);
	PFN_vkQueueWaitIdle pvkQueueWaitIdle = (PFN_vkQueueWaitIdle)vkGetDeviceProcAddr(device, "vkQueueWaitIdle");
	assert(pvkQueueWaitIdle);
	PFN_vkCmdBindDescriptorSets pvkCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)vkGetDeviceProcAddr(device, "vkCmdBindDescriptorSets");
	assert(pvkCmdBindDescriptorSets);
	PFN_vkCmdDispatch pvkCmdDispatch = (PFN_vkCmdDispatch)vkGetDeviceProcAddr(device, "vkCmdDispatch");
	assert(pvkCmdDispatch);
	PFN_vkCmdDraw pvkCmdDraw = (PFN_vkCmdDraw)vkGetDeviceProcAddr(device, "vkCmdDraw");
	assert(pvkCmdDraw);

	//main loop
	uint32_t index = 0;
	VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	VkCommandBuffer cmd{ VK_NULL_HANDLE };
	VkClearValue clearValues[1] = { {0.0f,0.0f,0.0f,0.0f} };
	VkRenderPassBeginInfo renderPassBeginInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
	renderPassBeginInfo.renderPass = renderPass;
	renderPassBeginInfo.renderArea = { 0,0,WIDTH,HEIGHT };
	renderPassBeginInfo.clearValueCount = sizeof(clearValues) / sizeof(clearValues[0]);
	renderPassBeginInfo.pClearValues = clearValues;
	VkDeviceSize offsets[1] = { 0 };
	VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo		submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.pWaitDstStageMask = &submitPipelineStages;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &presentComplete;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &renderComplete;
	submitInfo.commandBufferCount = 1;
	VkSubmitInfo computeInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	computeInfo.commandBufferCount = 1;
	
	VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapchain;
	presentInfo.pImageIndices = &index;
	presentInfo.pWaitSemaphores = &renderComplete;
	presentInfo.waitSemaphoreCount = 1;
	VkResult res;
	uint32_t frameCount = 0;
	while (!glfwWindowShouldClose(window)) {
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, 1);
		else if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
			saveScreenCap(device, commandBuffers[index], graphicsQueue, swapchainImages[index], memoryProperties, formatProperties, swapchainFormat.format, swapchainExtent, frameCount);
		}
		glfwPollEvents();
		uint32_t computeIdx = frameCount % swapchainImages.size();
		VkFence computeFence = computeFences[computeIdx];
		vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &computeFence);
		VkCommandBuffer computeCommandBuffer = computeCommandBuffers[computeIdx];
		computeInfo.pCommandBuffers = &computeCommandBuffer;
		pvkBeginCommandBuffer(computeCommandBuffer, &beginInfo);
		pvkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
		pvkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSet, 0, 0);
		pvkCmdDispatch(computeCommandBuffer, (uint32_t)(imageWidth / 16),(uint32_t)( imageHeight / 16), 1);
		pvkEndCommandBuffer(computeCommandBuffer);
		res = pvkQueueSubmit(computeQueue, 1, &computeInfo, computeFence);
		assert(res == VK_SUCCESS);
		
		res = pvkAcquireNextImage(device, swapchain, UINT64_MAX, presentComplete, nullptr, &index);
		VkFence currFence = renderFences[index];
		vkWaitForFences(device, 1, &currFence, VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &currFence);
		assert(res == VK_SUCCESS);

		cmd = commandBuffers[index];
		//transitionImage(device,graphicsQueue,cmd, computeImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
		pvkBeginCommandBuffer(cmd, &beginInfo);
		renderPassBeginInfo.framebuffer = framebuffers[index];
		pvkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		pvkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		pvkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, 0);
		pvkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer.buffer, offsets);
		pvkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		pvkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
		//pvkCmdDraw(cmd, 3, 1, 0, 0);
		pvkCmdEndRenderPass(cmd);

		res = pvkEndCommandBuffer(cmd);
		assert(res == VK_SUCCESS);

		submitInfo.pCommandBuffers = &cmd;
		res = pvkQueueSubmit(graphicsQueue, 1, &submitInfo, currFence);
		assert(res == VK_SUCCESS);
		res = pvkQueuePresent(presentQueue, &presentInfo);
		assert(res == VK_SUCCESS);
		pvkQueueWaitIdle(presentQueue);
		frameCount++;
	}

	vkDeviceWaitIdle(device);

	cleanupPipeline(device, pipeline);
	cleanupPipelineLayout(device, pipelineLayout);
	cleanupDescriptorPool(device, descriptorPool);
	cleanupDescriptorSetLayout(device, descriptorSetLayout);
	cleanupBuffer(device, indexBuffer);
	cleanupBuffer(device, vertexBuffer);
	cleanupFramebuffers(device, framebuffers);
	cleanupRenderPass(device, renderPass);
	
	

	cleanupPipeline(device, computePipeline);
	cleanupPipelineLayout(device, computePipelineLayout);
	//cleanupDescriptorSet(device,computeDescriptorPool, computeDescriptorSet);
	cleanupDescriptorPool(device, computeDescriptorPool);
	cleanupDescriptorSetLayout(device, computeDescriptorSetLayout);
	unmapBuffer(device, uboBuffer);
	cleanupBuffer(device, uboBuffer);
	cleanupImage(device, computeImage);
	for (auto& fence : computeFences) {
		cleanupFence(device, fence);
	}

	for (auto& fence : renderFences) {
		cleanupFence(device, fence);
	}

	
	cleanupCommandBuffers(device, computeCommandPool, computeCommandBuffers);
	cleanupCommandPool(device, computeCommandPool);
	cleanupCommandBuffers(device, commandPool, commandBuffers);
	cleanupCommandBuffer(device, commandPool, commandBuffer);
	cleanupCommandPool(device, commandPool);
	cleanupSemaphore(device, renderComplete);
	cleanupSemaphore(device, presentComplete);

	cleanupSwapchainImageViews(device, swapchainImageViews);
	cleanupSwapchain(device, swapchain);
	cleanupDevice(device);
	cleanupSurface(instance, surface);
	cleanupInstance(instance);
	cleanupWindow(window);

	return EXIT_SUCCESS;
}