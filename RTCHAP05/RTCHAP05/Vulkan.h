#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include <climits>
#include <stb_image_write.h>//handy for loading/saving bitmaps

// WINDOW SPECIFIC 
// Choose SDL2 or GLFW
//#define __USE__GLFW__

#if !defined __USE__GLFW__ && !defined __USE__SDL2__
#define __USE__GLFW__
#endif

#ifdef __USE__GLFW__
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

GLFWwindow* initWindow(const char*title,uint32_t width, uint32_t height);
VkSurfaceKHR initSurface(VkInstance instance, GLFWwindow* window);
void getRequiredInstanceExtensions(GLFWwindow* window, std::vector<const char*>& extensions);
void cleanupWindow(GLFWwindow* window);
#endif

#ifdef __USE__SDL2__
#include <SDL2/SDL.h>
#undef main
SDLWindow* initWindow(uint32_t width, uint32_t height);
VkSurfaceKHR initSurface(VkInstance instance, SDLWindow* window);

void getRequiredInstanceExtensions(SDLWindow* window, std::vector<string>& extensions);

#endif





VkInstance initInstance(std::vector<const char*>& requiredExtensions, std::vector<const char*>& requiredLayers);
void cleanupInstance(VkInstance instance);

void cleanupSurface(VkInstance instance, VkSurfaceKHR surface);

struct Queues {
	uint32_t graphicsQueueFamily;
	uint32_t presentQueueFamily;
	uint32_t computeQueueFamily;
};

VkPhysicalDevice choosePhysicalDevice(VkInstance instance, VkSurfaceKHR surface, Queues& queues);

VkDevice initDevice(VkPhysicalDevice physicalDevice, std::vector<const char*> deviceExtensions, Queues queues, VkPhysicalDeviceFeatures enabledFeatures);
void cleanupDevice(VkDevice device);

VkQueue getDeviceQueue(VkDevice device, uint32_t queueFamily);

VkPresentModeKHR chooseSwapchainPresentMode(std::vector<VkPresentModeKHR>& presentModes);

VkSurfaceFormatKHR chooseSwapchainFormat(std::vector<VkSurfaceFormatKHR>& formats);

VkSurfaceTransformFlagsKHR chooseSwapchainTransform(VkSurfaceCapabilitiesKHR& surfaceCaps);

VkCompositeAlphaFlagBitsKHR chooseSwapchainComposite(VkSurfaceCapabilitiesKHR& surfaceCaps);

VkSwapchainKHR initSwapchain(VkDevice device, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR& surfaceCaps, VkPresentModeKHR& presentMode, VkSurfaceFormatKHR& swapchainFormat, VkExtent2D& swapchainExtent, uint32_t& numImages);

void cleanupSwapchain(VkDevice device, VkSwapchainKHR swapchain);


void getSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, std::vector<VkImage>& images);

void initSwapchainImageViews(VkDevice device, std::vector<VkImage>& swapchainImages, VkFormat& swapchainFormat, std::vector<VkImageView>& swapchainImageViews);

void cleanupSwapchainImageViews(VkDevice device, std::vector<VkImageView>& imageViews);

VkSemaphore initSemaphore(VkDevice device);
void cleanupSemaphore(VkDevice device, VkSemaphore semaphore);

VkCommandPool initCommandPool(VkDevice device, uint32_t queueFamily);
void cleanupCommandPool(VkDevice device, VkCommandPool commandPool);

void initCommandPools(VkDevice device, size_t size, uint32_t queueFamily, std::vector<VkCommandPool>& commandPools);
void cleanupCommandPools(VkDevice device, std::vector<VkCommandPool>& commandPools);

VkCommandBuffer initCommandBuffer(VkDevice device, VkCommandPool commandPool);
void cleanupCommandBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer);


void initCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t count, std::vector<VkCommandBuffer>& commandBuffers);
void cleanupCommandBuffers(VkDevice device, VkCommandPool commandPool, std::vector<VkCommandBuffer>& commandBuffers);

VkRenderPass initRenderPass(VkDevice device, VkFormat colorFormat);
void cleanupRenderPass(VkDevice device, VkRenderPass renderPass);

void initFramebuffers(VkDevice device, VkRenderPass renderPass, std::vector<VkImageView>& colorAttachments, uint32_t width, uint32_t height, std::vector<VkFramebuffer>& framebuffers);
void cleanupFramebuffers(VkDevice device, std::vector<VkFramebuffer>& framebuffers);

struct ShaderModule {
	VkShaderModule shaderModule;
	VkShaderStageFlagBits stage;
};

VkShaderModule initShaderModule(VkDevice device, const char* filename);
void cleanupShaderModule(VkDevice device, VkShaderModule shaderModule);


struct Buffer {
	VkBuffer	buffer{ VK_NULL_HANDLE };
	VkDeviceMemory memory{ VK_NULL_HANDLE };
	VkDeviceSize size{ 0 };

};

void initBuffer(VkDevice device, VkPhysicalDeviceMemoryProperties& memoryProperties, VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, Buffer& buffer);
void* mapBuffer(VkDevice device, Buffer& buffer);
void unmapBuffer(VkDevice device, Buffer& buffer);
void CopyBufferTo(VkDevice device, VkQueue queue, VkCommandBuffer cmd, Buffer& src, Buffer& dst, VkDeviceSize size);
void cleanupBuffer(VkDevice device, Buffer& buffer);

struct Image {
	VkImage	image{ VK_NULL_HANDLE };
	VkDeviceMemory memory{ VK_NULL_HANDLE };
	VkSampler sampler{ VK_NULL_HANDLE };
	VkImageView imageView{ VK_NULL_HANDLE };
};

void initImage(VkDevice device, VkFormat format, VkFormatProperties& formatProperties, VkPhysicalDeviceMemoryProperties& memoryProperties, VkMemoryPropertyFlags memoryPropertyFlags, uint32_t width, uint32_t height, Image& image);

void cleanupImage(VkDevice device, Image& image);

void transitionImage(VkDevice device, VkQueue queue, VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);
void saveScreenCap(VkDevice device, VkCommandBuffer cmd, VkQueue queue, VkImage srcImage, VkPhysicalDeviceMemoryProperties& memoryProperties, VkFormatProperties& formatProperties, VkFormat colorFormat, VkExtent2D extent, uint32_t index);
VkFence initFence(VkDevice device, VkFenceCreateFlags flags = 0);

void cleanupFence(VkDevice device, VkFence fence);


VkDescriptorSetLayout initDescriptorSetLayout(VkDevice device, std::vector<VkDescriptorSetLayoutBinding>& descriptorBindings);

void cleanupDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout);


VkDescriptorPool initDescriptorPool(VkDevice device, std::vector<VkDescriptorPoolSize>& descriptorPoolSizes, uint32_t maxSets);

void cleanupDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool);


VkDescriptorSet initDescriptorSet(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, VkDescriptorPool descriptorPool);


void updateDescriptorSets(VkDevice device, std::vector<VkWriteDescriptorSet> descriptorWrites);


VkPipelineLayout initPipelineLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout);

void cleanupPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout);

VkPipeline initGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkPipelineLayout pipelineLayout, VkExtent2D extent, std::vector<ShaderModule>& shaders, VkVertexInputBindingDescription& bindingDescription, std::vector<VkVertexInputAttributeDescription>& attributeDescriptions);


VkPipeline initComputePipeline(VkDevice device, VkPipelineLayout pipelineLayout, ShaderModule& shader);

void cleanupPipeline(VkDevice device, VkPipeline pipeline);