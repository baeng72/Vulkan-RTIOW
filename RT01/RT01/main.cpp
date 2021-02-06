#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <cstdio>
#include <cassert>

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>


const int WIDTH = 800;
const int HEIGHT = 600;



GLFWwindow* initWindow(uint32_t width, uint32_t height) {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	GLFWwindow *window= glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);
	assert(window != nullptr);
	//glfwSetWindowUserPointer(window, this);
	//glfwSetFramebufferSizeCallback(window, frambebuffer_size_callback);
	//glfwSetKeyCallback(window, key_callback);
	return window;
}

VkInstance initInstance(std::vector<const char*>& requiredExtensions, std::vector<const char*>&requiredLayers) {
	VkInstance instance{ VK_NULL_HANDLE };
	if (std::find(requiredExtensions.begin(), requiredExtensions.end(), "VK_KHR_surface") == requiredExtensions.end())
		requiredExtensions.push_back("VK_KHR_surface");
	if (std::find(requiredExtensions.begin(), requiredExtensions.end(), "VK_KHR_win32_surface") == requiredExtensions.end())
		requiredExtensions.push_back("VK_KHR_win32_surface");
	VkInstanceCreateInfo instanceCI{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	instanceCI.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
	instanceCI.ppEnabledExtensionNames = requiredExtensions.data();
	instanceCI.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
	instanceCI.ppEnabledLayerNames = requiredLayers.data();
	assert(vkCreateInstance(&instanceCI, nullptr, &instance) == VK_SUCCESS);
	assert(instance != VK_NULL_HANDLE);

return instance;
}

VkSurfaceKHR initSurface(VkInstance instance, GLFWwindow* window) {
	VkSurfaceKHR surface{ VK_NULL_HANDLE };

	glfwCreateWindowSurface(instance, window, nullptr, &surface);
	assert(surface != VK_NULL_HANDLE);

	return surface;
}

struct Queues {
	uint32_t graphicsQueueFamily;
	uint32_t presentQueueFamily;
	uint32_t computeQueueFamily;
};

VkPhysicalDevice choosePhysicalDevice(VkInstance instance, VkSurfaceKHR surface, Queues& queues) {
	VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
	uint32_t physicalDeviceCount = 0;
	assert(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr) == VK_SUCCESS);
	assert(physicalDeviceCount > 0);
	std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
	assert(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data()) == VK_SUCCESS);


	for (size_t i = 0; i < physicalDevices.size(); i++) {
		VkPhysicalDevice phys = physicalDevices[i];
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(phys, &physicalDeviceProperties);

		if (physicalDeviceProperties.deviceType & VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			physicalDevice = physicalDevices[i];
			break;
		}
	}
	assert(physicalDevice != VK_NULL_HANDLE);
	uint32_t graphicsQueueFamily = UINT32_MAX;
	uint32_t presentQueueFamily = UINT32_MAX;
	uint32_t computeQueueFamily = UINT32_MAX;

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());


	for (uint32_t i = 0; i < queueFamilyCount; i++) {
		VkBool32 supportsPresent = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &supportsPresent);
		VkQueueFamilyProperties& queueProps = queueFamilyProperties[i];
		if (graphicsQueueFamily == UINT32_MAX && queueProps.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			graphicsQueueFamily = i;
		if (presentQueueFamily == UINT32_MAX && supportsPresent)
			presentQueueFamily = i;
		if (computeQueueFamily == UINT32_MAX && queueProps.queueFlags & VK_QUEUE_COMPUTE_BIT)
			computeQueueFamily = i;
		if (graphicsQueueFamily != UINT32_MAX && presentQueueFamily != UINT32_MAX && computeQueueFamily != UINT32_MAX)
			break;
	}
	assert(graphicsQueueFamily != UINT32_MAX && presentQueueFamily != UINT32_MAX && computeQueueFamily != UINT32_MAX);
	assert(computeQueueFamily == graphicsQueueFamily && graphicsQueueFamily == presentQueueFamily);//support one queue for now	
	queues.graphicsQueueFamily = graphicsQueueFamily;
	queues.presentQueueFamily = presentQueueFamily;
	queues.computeQueueFamily = computeQueueFamily;
	return physicalDevice;
}

VkDevice initDevice(VkPhysicalDevice physicalDevice, std::vector<const char*> deviceExtensions, Queues queues, VkPhysicalDeviceFeatures enabledFeatures) {
	VkDevice device{ VK_NULL_HANDLE };
	std::vector<float> queuePriorities;
	std::vector<VkDeviceQueueCreateInfo> queueCIs;

	if (queues.computeQueueFamily == queues.graphicsQueueFamily && queues.graphicsQueueFamily == queues.presentQueueFamily) {
		queuePriorities.push_back(1.0f);
		queueCIs.push_back({ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,nullptr,0,queues.graphicsQueueFamily,1,queuePriorities.data() });
	}
	else {
		//shouldn't get here for now
	}

	VkDeviceCreateInfo deviceCI{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	deviceCI.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	deviceCI.ppEnabledExtensionNames = deviceExtensions.data();
	deviceCI.pEnabledFeatures = &enabledFeatures;
	deviceCI.queueCreateInfoCount = static_cast<uint32_t>(queueCIs.size());
	deviceCI.pQueueCreateInfos = queueCIs.data();
	assert(vkCreateDevice(physicalDevice, &deviceCI, nullptr, &device) == VK_SUCCESS);
	return device;
}

VkQueue getDeviceQueue(VkDevice device, uint32_t queueFamily) {
	VkQueue queue{ VK_NULL_HANDLE };
	vkGetDeviceQueue(device, queueFamily, 0, &queue);
	assert(queue);
	return queue;
}


VkPresentModeKHR chooseSwapchainPresentMode(std::vector<VkPresentModeKHR>& presentModes) {
	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	for (size_t i = 0; i < presentModes.size(); i++) {
		if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
			presentMode = presentModes[i];
			break;
		}
		if ((presentMode != VK_PRESENT_MODE_MAILBOX_KHR) && (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)){
			presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
		}
	}
	return presentMode;


	return presentMode;
}

VkSurfaceFormatKHR chooseSwapchainFormat(std::vector<VkSurfaceFormatKHR>& formats) {
	VkSurfaceFormatKHR format;
	if (formats.size() >0)
		format = formats[0];
	
	for (auto&& surfaceFormat : formats) {
		if (surfaceFormat.format == VK_FORMAT_B8G8R8A8_UNORM) {
			format = surfaceFormat;
			break;
		}
	}
	return format;
}

VkSurfaceTransformFlagsKHR chooseSwapchainTransform(VkSurfaceCapabilitiesKHR& surfaceCaps) {

	VkSurfaceTransformFlagsKHR transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	if (!(surfaceCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR))
		transform = surfaceCaps.currentTransform;
	return transform;
}

VkCompositeAlphaFlagBitsKHR chooseSwapchainComposite(VkSurfaceCapabilitiesKHR& surfaceCaps) {
	VkCompositeAlphaFlagBitsKHR compositeFlags = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	std::vector<VkCompositeAlphaFlagBitsKHR> compositeAlphaFlags = {
			VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
			VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
			VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
	};
	for (auto& compositeAlphaFlag : compositeAlphaFlags) {
		if (surfaceCaps.supportedCompositeAlpha & compositeAlphaFlag) {
			compositeFlags = compositeAlphaFlag;
			break;
		};
	}
	return compositeFlags;
}

VkSwapchainKHR initSwapchain(VkDevice device,VkSurfaceKHR surface,uint32_t width,uint32_t height, VkSurfaceCapabilitiesKHR &surfaceCaps,VkPresentModeKHR&presentMode, VkSurfaceFormatKHR& swapchainFormat,VkExtent2D& swapchainExtent) {
	VkSwapchainKHR swapchain{ VK_NULL_HANDLE };

	VkSurfaceTransformFlagsKHR preTransform = chooseSwapchainTransform(surfaceCaps);
	VkCompositeAlphaFlagBitsKHR compositeAlpha = chooseSwapchainComposite(surfaceCaps);

	if (surfaceCaps.currentExtent.width == (uint32_t)-1) {
		swapchainExtent.width = width;
		swapchainExtent.height = height;
	}
	else {
		swapchainExtent = surfaceCaps.currentExtent;
	}

	uint32_t desiredNumberOfSwapchainImages = surfaceCaps.minImageCount + 1;
	if ((surfaceCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfaceCaps.maxImageCount))
	{
		desiredNumberOfSwapchainImages = surfaceCaps.maxImageCount;
	}

	VkSwapchainCreateInfoKHR swapchainCI = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
		
	swapchainCI.surface = surface;
	swapchainCI.minImageCount = desiredNumberOfSwapchainImages;
	swapchainCI.imageFormat = swapchainFormat.format;
	swapchainCI.imageColorSpace = swapchainFormat.colorSpace;
	swapchainCI.imageExtent = { swapchainExtent.width, swapchainExtent.height };
	swapchainCI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	swapchainCI.preTransform = (VkSurfaceTransformFlagBitsKHR)preTransform;
	swapchainCI.imageArrayLayers = 1;
	swapchainCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	swapchainCI.queueFamilyIndexCount = 0;
	swapchainCI.pQueueFamilyIndices = nullptr;
	swapchainCI.presentMode = presentMode;
	swapchainCI.oldSwapchain = VK_NULL_HANDLE;
	swapchainCI.clipped = VK_TRUE;
	swapchainCI.compositeAlpha = compositeAlpha;
	if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
		swapchainCI.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}

	// Enable transfer destination on swap chain images if supported
	if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
		swapchainCI.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}
	assert(vkCreateSwapchainKHR(device, &swapchainCI, nullptr, &swapchain) == VK_SUCCESS);		
	return swapchain;
}

void getSwapchainImages(VkDevice device,VkSwapchainKHR swapchain,std::vector<VkImage>& images) {
	uint32_t imageCount = 0;
	assert(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr) == VK_SUCCESS);
	images.resize(imageCount);
	assert(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data()) == VK_SUCCESS);
}

void initSwapchainImageViews(VkDevice device, std::vector<VkImage>& swapchainImages,VkFormat&swapchainFormat, std::vector<VkImageView>& swapchainImageViews) {
	swapchainImageViews.resize(swapchainImages.size());
	for (size_t i = 0; i < swapchainImages.size(); i++) {
		VkImageViewCreateInfo viewCI{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
		viewCI.format = swapchainFormat;
		viewCI.components = { VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY };
		viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;//attachment/view with be color
		viewCI.subresourceRange.baseMipLevel = 0;
		viewCI.subresourceRange.levelCount = 1;
		viewCI.subresourceRange.baseArrayLayer = 0;
		viewCI.subresourceRange.layerCount = 1;
		viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewCI.image = swapchainImages[i];
		assert(vkCreateImageView(device, &viewCI, nullptr, &swapchainImageViews[i]) == VK_SUCCESS);
	}
}

void cleanupSwapchainImageViews(VkDevice device, std::vector<VkImageView>& imageViews) {
	for (auto& imageView : imageViews) {
		vkDestroyImageView(device, imageView, nullptr);
	}
}

VkSemaphore initSemaphore(VkDevice device) {
	VkSemaphore semaphore{ VK_NULL_HANDLE };
	VkSemaphoreCreateInfo semaphoreCI{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	assert(vkCreateSemaphore(device, &semaphoreCI, nullptr, &semaphore) == VK_SUCCESS);
	return semaphore;
}


VkCommandPool initCommandPool(VkDevice device, uint32_t queueFamily) {
	VkCommandPool commandPool{ VK_NULL_HANDLE };
	VkCommandPoolCreateInfo cmdPoolCI{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	cmdPoolCI.queueFamilyIndex = queueFamily;
	cmdPoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	assert(vkCreateCommandPool(device, &cmdPoolCI, nullptr, &commandPool) == VK_SUCCESS);
	return commandPool;
}

void initCommandPools(VkDevice device, size_t size,uint32_t queueFamily, std::vector<VkCommandPool>& commandPools) {
	commandPools.resize(size, VK_NULL_HANDLE);
	for (size_t i = 0; i < size; i++) {
		VkCommandPoolCreateInfo cmdPoolCI{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
		cmdPoolCI.queueFamilyIndex = queueFamily;
		cmdPoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		assert(vkCreateCommandPool(device, &cmdPoolCI, nullptr, &commandPools[i]) == VK_SUCCESS);
	}
}

VkCommandBuffer initCommandBuffer(VkDevice device, VkCommandPool commandPool) {
	VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };
	VkCommandBufferAllocateInfo cmdBufAI{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	cmdBufAI.commandPool = commandPool;
	cmdBufAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAI.commandBufferCount = 1;
	assert(vkAllocateCommandBuffers(device, &cmdBufAI, &commandBuffer) == VK_SUCCESS);
		

	return commandBuffer;
}

void initCommandBuffers(VkDevice device, std::vector<VkCommandPool>& commandPools,std::vector<VkCommandBuffer>&commandBuffers) {
	commandBuffers.resize(commandPools.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < commandPools.size(); i++) {
		VkCommandBufferAllocateInfo cmdBufAI{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
		cmdBufAI.commandPool = commandPools[i];
		cmdBufAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAI.commandBufferCount = 1;
		assert(vkAllocateCommandBuffers(device, &cmdBufAI, &commandBuffers[i]) == VK_SUCCESS);
	}
}

VkRenderPass initRenderPass(VkDevice device, VkFormat colorFormat) {
	VkRenderPass renderPass{ VK_NULL_HANDLE };
	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = colorFormat;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorRef = { 0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorRef;

	VkSubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkRenderPassCreateInfo renderCI = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
	renderCI.attachmentCount = 1;
	renderCI.pAttachments = &colorAttachment;
	renderCI.subpassCount = 1;
	renderCI.pSubpasses = &subpass;
	renderCI.dependencyCount = 1;
	renderCI.pDependencies = &dependency;

	assert(vkCreateRenderPass(device, &renderCI, nullptr, &renderPass) == VK_SUCCESS);

	return renderPass;
}

void initFramebuffers(VkDevice device, VkRenderPass renderPass, std::vector<VkImageView>& colorAttachments, uint32_t width, uint32_t height, std::vector<VkFramebuffer>& framebuffers) {
	framebuffers.resize(colorAttachments.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < colorAttachments.size(); i++) {
		VkImageView attachments[] = { colorAttachments[i] };
		VkFramebufferCreateInfo fbCI{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
		fbCI.renderPass = renderPass;
		fbCI.attachmentCount = sizeof(attachments) / sizeof(attachments[0]);
		fbCI.pAttachments = attachments;
		fbCI.width = width;
		fbCI.height = height;
		fbCI.layers = 1;
		assert(vkCreateFramebuffer(device, &fbCI, nullptr, &framebuffers[i]) == VK_SUCCESS);
	}
}

struct ShaderModule {
	VkShaderModule shaderModule;
	VkShaderStageFlagBits stage;
};

VkShaderModule initShaderModule(VkDevice device, const char* filename) {
	VkShaderModule shaderModule{ VK_NULL_HANDLE };
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	createInfo.codeSize = buffer.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());
	assert(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) == VK_SUCCESS);
	return shaderModule;
}

struct Buffer{
	VkBuffer	buffer{ VK_NULL_HANDLE };
	VkDeviceMemory memory{ VK_NULL_HANDLE };
	VkDeviceSize size{ 0 };
	
};

uint32_t findMemoryType(uint32_t typeFilter, VkPhysicalDeviceMemoryProperties memoryProperties, VkMemoryPropertyFlags properties) {
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
		if (typeFilter & (1 << i) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
	assert(0);
	return 0;
}

VkFence initFence(VkDevice device,VkFenceCreateFlags flags=0) {
	VkFenceCreateInfo fenceCI{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	fenceCI.flags = flags;
	VkFence fence{ VK_NULL_HANDLE };
	assert(vkCreateFence(device, &fenceCI, nullptr, &fence) == VK_SUCCESS);
	return fence;
}


void initBuffer(VkDevice device, VkPhysicalDeviceMemoryProperties& memoryProperties, VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags,Buffer&buffer) {
	VkBufferCreateInfo bufferCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferCI.size = size;
	bufferCI.usage = usageFlags;
	bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	assert(vkCreateBuffer(device, &bufferCI, nullptr, &buffer.buffer) == VK_SUCCESS);
	VkMemoryRequirements memReqs{};
	vkGetBufferMemoryRequirements(device, buffer.buffer, &memReqs);
	VkMemoryAllocateInfo memAllocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	memAllocInfo.allocationSize = memReqs.size;
	memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, memoryProperties, memoryPropertyFlags);
	assert(vkAllocateMemory(device, &memAllocInfo, nullptr, &buffer.memory) == VK_SUCCESS);
	assert(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0) == VK_SUCCESS);
	buffer.size = size;
}

void* mapBuffer(VkDevice device,Buffer& buffer) {
	void* pData{ nullptr };
	assert(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &pData) == VK_SUCCESS);
	return pData;
}

void unmapBuffer(VkDevice device, Buffer& buffer) {
	vkUnmapMemory(device, buffer.memory);
}

void CopyBufferTo(VkDevice device,VkQueue queue, VkCommandBuffer cmd, Buffer& src, Buffer& dst,VkDeviceSize size) {
	VkBufferCopy copyRegion = {};
	VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };

	assert(vkBeginCommandBuffer(cmd, &beginInfo) == VK_SUCCESS);

	copyRegion.size = size;
	vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &copyRegion);

	assert(vkEndCommandBuffer(cmd) == VK_SUCCESS);

	VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;

	VkFence fence = initFence(device);
	

	assert(vkQueueSubmit(queue, 1, &submitInfo, fence) == VK_SUCCESS);
		
	assert(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS);
		

	vkDestroyFence(device, fence, nullptr);
}

struct Image {
	VkImage	image;
	VkDeviceMemory memory;
	VkSampler sampler;
	VkImageView imageView;
};

void initImage(VkDevice device,VkFormat format, VkFormatProperties &formatProperties, VkPhysicalDeviceMemoryProperties& memoryProperties, VkMemoryPropertyFlags memoryPropertyFlags,uint32_t width,uint32_t height, Image& image) {
	VkImageCreateInfo imageCI{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
	imageCI.imageType = VK_IMAGE_TYPE_2D;
	imageCI.format = format;
	imageCI.extent = { width,height,1 };
	imageCI.mipLevels = 1;
	imageCI.arrayLayers = 1;
	imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
	assert(vkCreateImage(device, &imageCI, nullptr, &image.image) == VK_SUCCESS);

	VkMemoryRequirements memReqs{};
	vkGetImageMemoryRequirements(device, image.image, &memReqs);
	VkMemoryAllocateInfo memAllocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	memAllocInfo.allocationSize = memReqs.size;
	memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, memoryProperties, memoryPropertyFlags);
	assert(vkAllocateMemory(device, &memAllocInfo, nullptr, &image.memory) == VK_SUCCESS);
	assert(vkBindImageMemory(device, image.image, image.memory, 0) == VK_SUCCESS);

	VkImageViewCreateInfo imageViewCI{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
	imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
	imageViewCI.format = format;
	imageViewCI.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
	imageViewCI.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1 };
	imageViewCI.image = image.image;
	assert(vkCreateImageView(device, &imageViewCI, nullptr, &image.imageView) == VK_SUCCESS);

	VkSamplerCreateInfo samplerCI{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	samplerCI.magFilter = samplerCI.minFilter = VK_FILTER_LINEAR;
	samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCI.addressModeU = samplerCI.addressModeV = samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samplerCI.mipLodBias = 0.0f;
	samplerCI.maxAnisotropy = 1.0f;
	samplerCI.compareOp = VK_COMPARE_OP_NEVER;
	samplerCI.minLod = samplerCI.maxLod = 0.0f;
	samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	assert(vkCreateSampler(device, &samplerCI, nullptr, &image.sampler) == VK_SUCCESS);
}


void transitionImage(VkDevice device,VkQueue queue,VkCommandBuffer cmd, Image& image, VkImageLayout oldLayout, VkImageLayout newLayout) {
	VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image.image;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	VkPipelineStageFlags sourceStage= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	VkPipelineStageFlags destinationStage= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else {
		
	}
	VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	assert(vkBeginCommandBuffer(cmd, &beginInfo)==VK_SUCCESS);
	vkCmdPipelineBarrier(cmd,sourceStage,destinationStage,
		0, 0, nullptr, 0, nullptr, 1, &barrier);
	assert(vkEndCommandBuffer(cmd)==VK_SUCCESS);

	VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;

	VkFence fence = initFence(device);


	assert(vkQueueSubmit(queue, 1, &submitInfo, fence) == VK_SUCCESS);

	assert(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS);


	vkDestroyFence(device, fence, nullptr);
}

VkDescriptorSetLayout initDescriptorSetLayout(VkDevice device, std::vector<VkDescriptorSetLayoutBinding>& descriptorBindings) {
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
	VkDescriptorSetLayoutCreateInfo descLayoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	descLayoutCI.bindingCount = static_cast<uint32_t>(descriptorBindings.size());
	descLayoutCI.pBindings = descriptorBindings.data();
	assert(vkCreateDescriptorSetLayout(device, &descLayoutCI, nullptr, &descriptorSetLayout) == VK_SUCCESS);		
	return descriptorSetLayout;
}

VkDescriptorPool initDescriptorPool(VkDevice device, std::vector<VkDescriptorPoolSize>& descriptorPoolSizes, uint32_t maxSets) {
	VkDescriptorPool descriptorPool{ VK_NULL_HANDLE };
	VkDescriptorPoolCreateInfo descPoolCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	descPoolCI.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());
	descPoolCI.pPoolSizes = descriptorPoolSizes.data();
	descPoolCI.maxSets = maxSets;
	assert(vkCreateDescriptorPool(device, &descPoolCI, nullptr, &descriptorPool) == VK_SUCCESS);
	return descriptorPool;
}

VkDescriptorSet initDescriptorSet(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, VkDescriptorPool descriptorPool) {
	VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
	VkDescriptorSetAllocateInfo descAI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	descAI.descriptorPool = descriptorPool;
	descAI.descriptorSetCount = 1;
	descAI.pSetLayouts = &descriptorSetLayout;

	assert(vkAllocateDescriptorSets(device, &descAI, &descriptorSet) == VK_SUCCESS);
	return descriptorSet;
}

void updateDescriptorSets(VkDevice device, std::vector<VkWriteDescriptorSet> descriptorWrites) {
	vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(),0,nullptr);
}

VkPipelineLayout initPipelineLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout) {
	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkPipelineLayoutCreateInfo layoutCI{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	layoutCI.pSetLayouts = descriptorSetLayout == VK_NULL_HANDLE ? nullptr : &descriptorSetLayout;
	layoutCI.setLayoutCount = descriptorSetLayout == VK_NULL_HANDLE ? 0 : 1;
	assert(vkCreatePipelineLayout(device, &layoutCI, nullptr, &pipelineLayout) == VK_SUCCESS);
	return pipelineLayout;
}

VkPipeline initGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkPipelineLayout pipelineLayout,VkExtent2D extent,std::vector<ShaderModule>& shaders, VkVertexInputBindingDescription& bindingDescription, std::vector<VkVertexInputAttributeDescription>& attributeDescriptions) {
	VkPipeline pipeline{ VK_NULL_HANDLE };

	//we're working with triangles;
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	// Specify rasterization state. 
	VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = VK_CULL_MODE_BACK_BIT;// VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_CLOCKWISE;//  VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.lineWidth = 1.0f;

	//all colors, no blending
	VkPipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo blend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
	blend.attachmentCount = 1;
	blend.logicOpEnable = VK_FALSE;
	blend.logicOp = VK_LOGIC_OP_COPY;
	blend.pAttachments = &blendAttachment;

	//viewport & scissor box
	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)extent.width;
	viewport.height = (float)extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = extent;
	VkPipelineViewportStateCreateInfo viewportCI{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
	viewportCI.viewportCount = 1;
	viewportCI.pViewports = &viewport;
	viewportCI.scissorCount = 1;
	viewportCI.pScissors = &scissor;

	VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
	depthStencil.depthTestEnable = VK_FALSE;
	depthStencil.depthWriteEnable = VK_FALSE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;

	VkPipelineMultisampleStateCreateInfo multisamplingCI{};
	multisamplingCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisamplingCI.sampleShadingEnable = VK_FALSE;
	multisamplingCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisamplingCI.minSampleShading = 1.0f; // Optional
	multisamplingCI.pSampleMask = nullptr; // Optional
	multisamplingCI.alphaToCoverageEnable = VK_FALSE; // Optional
	multisamplingCI.alphaToOneEnable = VK_FALSE; // Optional

	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
	for (auto& shaderModule : shaders) {
		VkPipelineShaderStageCreateInfo shaderCI{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		shaderCI.stage = shaderModule.stage;
		shaderCI.module = shaderModule.shaderModule;
		shaderCI.pName = "main";
		shaderStages.push_back(shaderCI);
	}

	VkPipelineVertexInputStateCreateInfo vertexCI{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
	vertexCI.vertexBindingDescriptionCount = 1;
	vertexCI.pVertexBindingDescriptions = &bindingDescription;
	vertexCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexCI.pVertexAttributeDescriptions = attributeDescriptions.data();

	VkGraphicsPipelineCreateInfo pipe{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
	pipe.stageCount = static_cast<uint32_t>(shaderStages.size());
	pipe.pStages = shaderStages.data();
	pipe.pVertexInputState = &vertexCI;
	pipe.pInputAssemblyState = &inputAssembly;
	pipe.pRasterizationState = &raster;
	pipe.pColorBlendState = &blend;
	pipe.pMultisampleState = &multisamplingCI;
	pipe.pViewportState = &viewportCI;
	pipe.pDepthStencilState = nullptr;
	pipe.pDynamicState = nullptr;

	pipe.renderPass = renderPass;
	pipe.layout = pipelineLayout;

	assert(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipe, nullptr, &pipeline) == VK_SUCCESS);

	return pipeline;

}

VkPipeline initComputePipeline(VkDevice device, VkPipelineLayout pipelineLayout, ShaderModule& shader) {
	VkPipeline pipeline{ VK_NULL_HANDLE };
	VkPipelineShaderStageCreateInfo shaderCI{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	shaderCI.module = shader.shaderModule;
	shaderCI.pName = "main";
	shaderCI.stage = shader.stage;
	VkComputePipelineCreateInfo computeCI{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	computeCI.stage = shaderCI;
	computeCI.layout = pipelineLayout;
	assert(vkCreateComputePipelines(device, nullptr, 1, &computeCI, nullptr, &pipeline) == VK_SUCCESS);

	return pipeline;
}

void cleanupPipeline(VkDevice device, VkPipeline pipeline) {
	vkDestroyPipeline(device, pipeline, nullptr);
}

void cleanupPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout) {
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}


//void cleanupDescriptorSet(VkDevice device,VkDescriptorPool descriptorPool, VkDescriptorSet descriptorSet) {
//	vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
//}

void cleanupDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool) {
	vkDestroyDescriptorPool(device, descriptorPool, nullptr);
}


void cleanupDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout) {
	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void cleanupImage(VkDevice device, Image& image) {
	vkDestroySampler(device, image.sampler, nullptr);
	vkDestroyImageView(device, image.imageView, nullptr);
	vkFreeMemory(device, image.memory, nullptr);
	vkDestroyImage(device, image.image, nullptr);
}

void cleanupBuffer(VkDevice device, Buffer& buffer) {
	vkFreeMemory(device, buffer.memory, nullptr);
	vkDestroyBuffer(device, buffer.buffer, nullptr);

}

void cleanupFence(VkDevice device, VkFence fence) {
	vkDestroyFence(device, fence, nullptr);
}
void cleanupShaderModule(VkDevice device, VkShaderModule shaderModule) {
	vkDestroyShaderModule(device, shaderModule, nullptr);
}

void cleanupFramebuffers(VkDevice device, std::vector<VkFramebuffer>& framebuffers) {
	for (auto& framebuffer : framebuffers) {
		vkDestroyFramebuffer(device, framebuffer, nullptr);
	}
}

void cleanupRenderPass(VkDevice device, VkRenderPass renderPass) {
	vkDestroyRenderPass(device, renderPass, nullptr);
}



void cleanupCommandBuffers(VkDevice device, std::vector<VkCommandPool>& commandPools, std::vector<VkCommandBuffer>& commandBuffers) {
	for (size_t i = 0; i < commandBuffers.size(); i++) {
		vkFreeCommandBuffers(device, commandPools[i], 1, &commandBuffers[i]);
	}
}

void cleanupCommandBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer) {
	std::vector<VkCommandPool> commandPools{ commandPool };
	std::vector<VkCommandBuffer> commandBuffers{ commandBuffer };
	cleanupCommandBuffers(device, commandPools, commandBuffers);
}



void cleanupCommandPools(VkDevice device, std::vector<VkCommandPool>& commandPools) {
	for (auto& commandPool : commandPools) {
		vkDestroyCommandPool(device, commandPool, nullptr);
	}
}

void cleanupCommandPool(VkDevice device, VkCommandPool commandPool) {
	std::vector<VkCommandPool> commandPools{ commandPool };
	cleanupCommandPools(device, commandPools);
}

void cleanupSemaphore(VkDevice device, VkSemaphore semaphore) {
	vkDestroySemaphore(device, semaphore, nullptr);
}
void cleanupSwapchain(VkDevice device, VkSwapchainKHR swapchain) {
	vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void cleanupDevice(VkDevice device) {
	vkDestroyDevice(device, nullptr);
}

void cleanupSurface(VkInstance instance,VkSurfaceKHR surface) {
	vkDestroySurfaceKHR(instance, surface, nullptr);
}

void cleanupInstance(VkInstance instance) {
	vkDestroyInstance(instance, nullptr);
}

void cleanupWindow(GLFWwindow* window) {
	glfwDestroyWindow(window);
	glfwTerminate();
}



int main() {
	GLFWwindow* window = initWindow(WIDTH, HEIGHT);


	uint32_t extCount = 0;
	auto ext = glfwGetRequiredInstanceExtensions(&extCount);
	std::vector<const char*> requiredExtensions(ext, ext + extCount);
	std::vector<const char*> requiredLayers{ "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor" };

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
	VkExtent2D swapchainExtent{};
	VkSwapchainKHR swapchain = initSwapchain(device, surface, WIDTH, HEIGHT, surfaceCaps, presentMode, swapchainFormat, swapchainExtent);
	std::vector<VkImage> swapchainImages;
	getSwapchainImages(device, swapchain, swapchainImages);
	std::vector<VkImageView> swapchainImageViews;
	initSwapchainImageViews(device, swapchainImages, swapchainFormat.format, swapchainImageViews);

	VkSemaphore presentComplete = initSemaphore(device);
	VkSemaphore renderComplete = initSemaphore(device);

	VkCommandPool commandPool = initCommandPool(device, queues.graphicsQueueFamily);
	VkCommandBuffer commandBuffer = initCommandBuffer(device, commandPool);

	VkCommandPool computeCommandPool = initCommandPool(device, queues.computeQueueFamily);
	VkCommandBuffer computeCommandBuffer = initCommandBuffer(device, computeCommandPool);
	VkFence computeFence = initFence(device,VK_FENCE_CREATE_SIGNALED_BIT);

	Image computeImage;
	initImage(device, swapchainFormat.format, formatProperties, memoryProperties, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1024, 1024, computeImage);
	transitionImage(device,graphicsQueue,commandBuffer, computeImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

	struct {
		float imageWidth;
		float imageHeight;
		float viewportWidth;
		float viewportHeight;
		float focalLength;
	}ubo;
	ubo.imageWidth = 1024.0f;
	ubo.imageHeight = 1024.0f;
	ubo.viewportHeight = 2.0f;
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
	VkShaderModule compShader = initShaderModule(device, "Shaders/raytrace3.comp.spv");	
	ShaderModule shader =  {compShader,VK_SHADER_STAGE_COMPUTE_BIT};
	VkPipeline computePipeline = initComputePipeline(device, computePipelineLayout, shader);
	cleanupShaderModule(device,compShader);
	std::vector<VkCommandPool> commandPools;
	initCommandPools(device, swapchainImages.size(), queues.graphicsQueueFamily, commandPools);

	std::vector<VkCommandBuffer> commandBuffers;
	initCommandBuffers(device, commandPools, commandBuffers);

	VkRenderPass renderPass = initRenderPass(device, swapchainFormat.format);

	std::vector<VkFramebuffer> framebuffers;
	initFramebuffers(device, renderPass, swapchainImageViews, WIDTH, HEIGHT, framebuffers);


	float vertices[] = {
		-1.0f,-1.0f,0.0f,0.0f,0.0f,//top left
		1.0f,-1.0f,0.0f,1.0f,0.0f,//top right
		1.0f,1.0f,0.0f,1.0f,1.0f,//bottom right
		-1.0f,1.0f,0.0f,0.0f,1.0f//bottom left;
	};
	//float vertices[] = {
	//	//pos				//color
	//	 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom right
	//	-0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom left
	//	 0.0f,  -0.5f, 0.0f, 0.0f, 0.0f, 1.0f   // top 
	//};
	//unsigned int indices[] = {
	//	0,1,2
	//};
	uint32_t indices[] = {
		3,0,1,3,1,2
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
	uint32_t index = UINT32_MAX;
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
	computeInfo.pCommandBuffers = &computeCommandBuffer;
	VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapchain;
	presentInfo.pImageIndices = &index;
	presentInfo.pWaitSemaphores = &renderComplete;
	presentInfo.waitSemaphoreCount = 1;
	while (!glfwWindowShouldClose(window)) {
		if (glfwGetKey(window, GLFW_KEY_ESCAPE))
			glfwSetWindowShouldClose(window, 1);
		glfwPollEvents();
		
		pvkBeginCommandBuffer(computeCommandBuffer, &beginInfo);
		pvkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
		pvkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSet, 0, 0);
		pvkCmdDispatch(computeCommandBuffer, 1024 / 16, 1024 / 16, 1);
		pvkEndCommandBuffer(computeCommandBuffer);
		assert(pvkQueueSubmit(computeQueue, 1, &computeInfo, VK_NULL_HANDLE) == VK_SUCCESS);
		vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &computeFence);
		assert(pvkAcquireNextImage(device, swapchain, UINT64_MAX, presentComplete, nullptr, &index) == VK_SUCCESS);
		
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

		assert(pvkEndCommandBuffer(cmd) == VK_SUCCESS);

		submitInfo.pCommandBuffers = &cmd;
		assert(pvkQueueSubmit(graphicsQueue, 1, &submitInfo, computeFence)==VK_SUCCESS);
		assert(pvkQueuePresent(presentQueue, &presentInfo) == VK_SUCCESS);
		pvkQueueWaitIdle(presentQueue);

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
	cleanupCommandBuffers(device, commandPools, commandBuffers);
	cleanupCommandPools(device, commandPools);

	cleanupPipeline(device, computePipeline);
	cleanupPipelineLayout(device, computePipelineLayout);
	//cleanupDescriptorSet(device,computeDescriptorPool, computeDescriptorSet);
	cleanupDescriptorPool(device, computeDescriptorPool);
	cleanupDescriptorSetLayout(device, computeDescriptorSetLayout);
	unmapBuffer(device,uboBuffer);
	cleanupBuffer(device,uboBuffer);
	cleanupImage(device, computeImage);
	cleanupFence(device, computeFence);
	

	cleanupCommandBuffer(device,computeCommandPool, computeCommandBuffer);
	cleanupCommandPool(device, computeCommandPool);
	cleanupCommandBuffer(device,  commandPool ,  commandBuffer );
	cleanupCommandPool(device, commandPool);
	cleanupSemaphore(device, renderComplete);
	cleanupSemaphore(device, presentComplete);

	cleanupSwapchainImageViews(device, swapchainImageViews);
	cleanupSwapchain(device,swapchain);
	cleanupDevice(device);
	cleanupSurface(instance, surface);
	cleanupInstance(instance);
	cleanupWindow(window);

	return EXIT_SUCCESS;
}