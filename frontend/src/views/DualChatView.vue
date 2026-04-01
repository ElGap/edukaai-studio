<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="border-b border-slate-800 pb-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-white">Step 5: Dual Chat</h1>
          <p class="mt-2 text-slate-400">
            Test and compare your models. Select which model(s) to chat with below.
          </p>
        </div>
        
        <!-- Chat Mode Selector -->
        <div class="flex items-center gap-2 bg-slate-800 p-1 rounded-lg">
          <button
            @click="chatMode = 'both'"
            :class="[
              'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
              chatMode === 'both' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
            ]"
          >
            Both
          </button>
          <button
            @click="chatMode = 'base'"
            :class="[
              'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
              chatMode === 'base' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
            ]"
          >
            Base Only
          </button>
          <button
            @click="chatMode = 'finetuned'"
            :class="[
              'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
              chatMode === 'finetuned' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
            ]"
          >
            Fine-tuned Only
          </button>
        </div>
      </div>
    </div>

    <!-- Model Loading Status -->
    <div v-if="!modelsLoaded" class="bg-slate-800/50 rounded-xl p-8 text-center">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
      <p class="text-white font-medium">Loading models for inference...</p>
      <p class="text-slate-400 text-sm mt-2">This may take a few moments</p>
    </div>

    <div v-else :class="chatMode === 'both' ? 'grid gap-6 lg:grid-cols-2' : 'max-w-4xl mx-auto'">
      <!-- Base Model Panel -->
      <div v-if="chatMode !== 'finetuned'" class="bg-slate-900 rounded-xl border border-slate-800 flex flex-col h-[calc(100vh-820px)]">
        <!-- Panel Header -->
        <div class="p-4 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h2 class="font-semibold text-white">Base Model</h2>
            <p class="text-xs text-slate-400">{{ baseModelName }} (Original)</p>
          </div>
          <button
            @click="clearLeftChat"
            class="text-xs text-slate-400 hover:text-white transition-colors"
          >
            Clear
          </button>
        </div>

        <!-- Controls -->
        <div class="p-4 border-b border-slate-800 space-y-4">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">System Prompt</label>
            <textarea
              v-model="leftConfig.systemPrompt"
              rows="2"
              class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-blue-500 focus:outline-none resize-none"
            ></textarea>
          </div>
          
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs font-medium text-slate-400 mb-1">Temperature</label>
              <input
                v-model.number="leftConfig.temperature"
                type="range"
                min="0"
                max="2"
                step="0.1"
                class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
              <div class="flex justify-between text-xs text-slate-500 mt-1">
                <span>0</span>
                <span>{{ leftConfig.temperature }}</span>
                <span>2</span>
              </div>
            </div>
            
            <div>
              <label class="block text-xs font-medium text-slate-400 mb-1">Max Tokens</label>
              <select
                v-model.number="leftConfig.maxTokens"
                class="w-full px-2 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option :value="128">128</option>
                <option :value="256">256</option>
                <option :value="512">512 (Default)</option>
                <option :value="1024">1024</option>
                <option :value="2048">2048</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Chat History -->
        <div class="flex-1 overflow-y-auto p-4 space-y-4">
          <div
            v-for="(msg, index) in leftChatHistory"
            :key="index"
            :class="[
              'p-3 rounded-lg',
              msg.role === 'user' ? 'bg-slate-800 ml-8' : 'bg-blue-900/20 mr-8'
            ]"
          >
            <div class="formatted-message text-sm text-white" v-html="formatMessageContent(msg.content, msg.role as 'user' | 'assistant')"></div>
            <div v-if="msg.metrics" class="mt-2 pt-2 border-t border-slate-700/50 flex flex-wrap gap-2 text-xs text-slate-400">
              <span>{{ msg.metrics.responseTime }}s</span>
              <span>{{ msg.metrics.tokens }} tokens</span>
              <span>{{ msg.metrics.speed }} tok/s</span>
            </div>
          </div>
          
          <div v-if="isLeftGenerating" class="flex items-center gap-2 text-slate-400 text-sm">
            <span class="animate-pulse">Generating...</span>
          </div>
        </div>
      </div>

      <!-- Fine-Tuned Model Panel -->
      <div v-if="chatMode !== 'base'" class="bg-slate-900 rounded-xl border border-slate-800 flex flex-col h-[calc(100vh-820px)]">
        <!-- Panel Header -->
        <div class="p-4 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h2 class="font-semibold text-white">Fine-Tuned Model</h2>
            <p class="text-xs text-slate-400">{{ fineTunedModelName }}</p>
          </div>
          <div class="flex items-center gap-2">
            <button
              @click="openEditModal"
              class="text-xs text-slate-400 hover:text-white transition-colors"
            >
              Edit
            </button>
            <button
              @click="clearRightChat"
              class="text-xs text-slate-400 hover:text-white transition-colors"
            >
              Clear
            </button>
          </div>
        </div>

        <!-- Controls -->
        <div class="p-4 border-b border-slate-800 space-y-4">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">System Prompt</label>
            <textarea
              v-model="rightConfig.systemPrompt"
              rows="2"
              class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-blue-500 focus:outline-none resize-none"
            ></textarea>
          </div>
          
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs font-medium text-slate-400 mb-1">Temperature</label>
              <input
                v-model.number="rightConfig.temperature"
                type="range"
                min="0"
                max="2"
                step="0.1"
                class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
              <div class="flex justify-between text-xs text-slate-500 mt-1">
                <span>0</span>
                <span>{{ rightConfig.temperature }}</span>
                <span>2</span>
              </div>
            </div>
            
            <div>
              <label class="block text-xs font-medium text-slate-400 mb-1">Max Tokens</label>
              <select
                v-model.number="rightConfig.maxTokens"
                class="w-full px-2 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option :value="128">128</option>
                <option :value="256">256</option>
                <option :value="512">512 (Default)</option>
                <option :value="1024">1024</option>
                <option :value="2048">2048</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Chat History -->
        <div class="flex-1 overflow-y-auto p-4 space-y-4">
          <div
            v-for="(msg, index) in rightChatHistory"
            :key="index"
            :class="[
              'p-3 rounded-lg',
              msg.role === 'user' ? 'bg-slate-800 ml-8' : 'bg-green-900/20 mr-8'
            ]"
          >
            <div class="formatted-message text-sm text-white" v-html="formatMessageContent(msg.content, msg.role as 'user' | 'assistant')"></div>
            <div v-if="msg.metrics" class="mt-2 pt-2 border-t border-slate-700/50 flex flex-wrap gap-2 text-xs text-slate-400">
              <span>{{ msg.metrics.responseTime }}s</span>
              <span>{{ msg.metrics.tokens }} tokens</span>
              <span>{{ msg.metrics.speed }} tok/s</span>
            </div>
          </div>
          
          <div v-if="isRightGenerating" class="flex items-center gap-2 text-slate-400 text-sm">
            <span class="animate-pulse">Generating...</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Comparison Stats -->
    <div v-if="chatMode === 'both' && hasComparisonData" class="bg-slate-900 rounded-xl border border-slate-800 p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Comparison Statistics</h3>

      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
          <tr class="border-b border-slate-700">
            <th class="py-2 text-left text-slate-400">Metric</th>
            <th class="py-2 text-center text-slate-400">Base Model</th>
            <th class="py-2 text-center text-slate-400">Fine-Tuned</th>
            <th class="py-2 text-center text-slate-400">Difference</th>
          </tr>
          </thead>
          <tbody>
          <tr class="border-b border-slate-800">
            <td class="py-3 text-slate-400">Avg Response Time</td>
            <td class="py-3 text-center text-white">{{ leftStats.avgResponseTime }}s</td>
            <td class="py-3 text-center text-white">{{ rightStats.avgResponseTime }}s</td>
            <td class="py-3 text-center" :class="responseTimeDiff > 0 ? 'text-red-400' : 'text-green-400'">
              {{ responseTimeDiff > 0 ? '+' : '' }}{{ responseTimeDiff }}%
            </td>
          </tr>
          <tr class="border-b border-slate-800">
            <td class="py-3 text-slate-400">Avg Tokens Generated</td>
            <td class="py-3 text-center text-white">{{ leftStats.avgTokens }}</td>
            <td class="py-3 text-center text-white">{{ rightStats.avgTokens }}</td>
            <td class="py-3 text-center" :class="tokensDiff > 0 ? 'text-green-400' : 'text-red-400'">
              {{ tokensDiff > 0 ? '+' : '' }}{{ tokensDiff }}%
            </td>
          </tr>
          <tr>
            <td class="py-3 text-slate-400">Avg Speed</td>
            <td class="py-3 text-center text-white">{{ leftStats.avgSpeed }} tok/s</td>
            <td class="py-3 text-center text-white">{{ rightStats.avgSpeed }} tok/s</td>
            <td class="py-3 text-center" :class="speedDiff > 0 ? 'text-green-400' : 'text-red-400'">
              {{ speedDiff > 0 ? '+' : '' }}{{ speedDiff }}%
            </td>
          </tr>
          </tbody>
        </table>
      </div>
      <button
          @click="resetStats"
          class="mt-4 text-xs text-slate-400 hover:text-white transition-colors"
      >
        Reset Comparison Stats
      </button>
    </div>

    <!-- Shared Input -->
    <div class="bg-slate-900 rounded-xl border border-slate-800 p-4">
      <div class="flex gap-3">
        <input
          v-model="userInput"
          @keyup.enter="sendMessage"
          type="text"
          :placeholder="chatMode === 'both' ? 'Type a message to test both models...' : chatMode === 'base' ? 'Type a message for the base model...' : 'Type a message for the fine-tuned model...'"
          class="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
          :disabled="!modelsLoaded || isLeftGenerating || isRightGenerating"
        />
        <button
          @click="sendMessage"
          :disabled="!modelsLoaded || isLeftGenerating || isRightGenerating || !userInput.trim()"
          class="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg font-medium transition-colors"
        >
          {{ isLeftGenerating || isRightGenerating ? 'Generating...' : sendButtonText }}
        </button>
      </div>
    </div>





    <!-- Edit Model Modal -->
    <div
      v-if="showEditModal"
      class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      @click="showEditModal = false"
    >
      <div
        class="bg-slate-900 rounded-xl border border-slate-800 p-6 w-full max-w-lg mx-4"
        @click.stop
      >
        <h3 class="text-lg font-semibold text-white mb-4">Edit Model Metadata</h3>
        
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-slate-400 mb-1">Model Name</label>
            <input
              v-model="editName"
              type="text"
              class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              placeholder="Enter model name"
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-slate-400 mb-1">
              Description
              <span class="text-xs text-slate-500 ml-1">({{ editDescription.length }}/5000)</span>
            </label>
            <textarea
              v-model="editDescription"
              rows="4"
              maxlength="5000"
              class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none resize-none"
              placeholder="Enter model description"
            ></textarea>
          </div>
          
          <div>
            <label class="block text-sm font-medium text-slate-400 mb-1">Tags (comma-separated)</label>
            <input
              v-model="editTags"
              type="text"
              class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              placeholder="e.g., llm, chatbot, mlx"
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-slate-400 mb-1">
              Notes & Findings
              <span class="text-xs text-slate-500 ml-1">(your thoughts on fine-tuning)</span>
            </label>
            <textarea
              v-model="editNotes"
              rows="6"
              maxlength="10000"
              class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none resize-none"
              placeholder="Document your fine-tuning observations, what worked, what didn't, insights gained..."
            ></textarea>
            <p class="text-xs text-slate-500 mt-1 text-right">{{ editNotes.length }}/10000</p>
          </div>
        </div>
        
        <div class="flex justify-end gap-3 mt-6">
          <button
            @click="showEditModal = false"
            class="px-4 py-2 text-slate-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            @click="saveModelChanges"
            :disabled="isSaving || !editName.trim()"
            class="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg font-medium transition-colors"
          >
            {{ isSaving ? 'Saving...' : 'Save Changes' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useTrainingStore } from '@/stores/training'
import axios from 'axios'
import { useToast } from '@/composables/useToast'
import { formatMessage, sanitizeInput, validateInput } from '@/utils/messageSecurity'

const router = useRouter()
const store = useTrainingStore()
const { success: showSuccess, error: showError } = useToast()

const api = axios.create({
  baseURL: '/api'
})

// Chat mode
const chatMode = ref<'both' | 'base' | 'finetuned'>('both')

// Config state
const leftConfig = ref({
  systemPrompt: 'You are a helpful assistant.',
  temperature: 0.7,
  maxTokens: 512,
  topP: 0.9
})

const rightConfig = ref({
  systemPrompt: 'You are a helpful assistant.',
  temperature: 0.7,
  maxTokens: 512,
  topP: 0.9
})

// Chat history
const leftChatHistory = ref<Array<{ role: string; content: string; metrics?: any }>>([])
const rightChatHistory = ref<Array<{ role: string; content: string; metrics?: any }>>([])
const isLeftGenerating = ref(false)
const isRightGenerating = ref(false)
const userInput = ref('')
const modelsLoaded = ref(false)

// Edit modal state for fine-tuned model
const showEditModal = ref(false)
const editName = ref('')
const editDescription = ref('')
const editTags = ref('')
const editNotes = ref('')
const isSaving = ref(false)

// Model names
const baseModelName = computed(() => store.completedRun?.base_model?.name || 'Unknown Model')
const fineTunedModelName = computed(() => store.completedRun?.name || 'Fine-Tuned Model')
const runId = computed(() => store.completedRun?.id)

// Stats
const leftStats = ref({ avgResponseTime: 0, avgTokens: 0, avgSpeed: 0, count: 0 })
const rightStats = ref({ avgResponseTime: 0, avgTokens: 0, avgSpeed: 0, count: 0 })

const hasComparisonData = computed(() => leftChatHistory.value.length > 0 && rightChatHistory.value.length > 0)

const responseTimeDiff = computed(() => {
  if (leftStats.value.avgResponseTime === 0) return 0
  const diff = ((rightStats.value.avgResponseTime - leftStats.value.avgResponseTime) / leftStats.value.avgResponseTime * 100)
  return Number(diff.toFixed(1))
})

const sendButtonText = computed(() => {
  if (chatMode.value === 'both') return 'Send to Both'
  if (chatMode.value === 'base') return 'Send to Base'
  return 'Send to Fine-tuned'
})

const tokensDiff = computed(() => {
  if (leftStats.value.avgTokens === 0) return 0
  const diff = ((rightStats.value.avgTokens - leftStats.value.avgTokens) / leftStats.value.avgTokens * 100)
  return Number(diff.toFixed(1))
})

const speedDiff = computed(() => {
  if (leftStats.value.avgSpeed === 0) return 0
  const diff = ((rightStats.value.avgSpeed - leftStats.value.avgSpeed) / leftStats.value.avgSpeed * 100)
  return Number(diff.toFixed(1))
})

// Load models on mount
onMounted(async () => {
  let currentRunId = runId.value
  
  // If runId is not in store, try to get it from URL query params
  if (!currentRunId) {
    const urlParams = new URLSearchParams(window.location.search)
    currentRunId = urlParams.get('run_id') || ''
  }
  
  // If still no runId, check if we can fetch from the most recent completed run
  if (!currentRunId) {
    // Try to fetch the latest completed run
    try {
      const response = await api.get('/training/runs?status=completed')
      if (response.data && response.data.length > 0) {
        // Get the most recent completed run
        const latestRun = response.data[0]
        store.setCompletedRun(latestRun)
        currentRunId = latestRun.id
      }
    } catch (err) {
      console.error('Failed to fetch training runs:', err)
    }
  }
  
  if (!currentRunId) {
    showError('No completed training run found')
    router.push({ name: 'summary' })
    return
  }
  
  // Ensure we have the latest run data with description and tags
  try {
    await store.fetchCompletedRun(currentRunId)
  } catch (err) {
    console.error('Failed to fetch run details:', err)
  }
  
  try {
    showSuccess('Loading models for chat...')
    
    // Load base model
    await api.post('/chat/load-model', {
      run_id: currentRunId,
      use_fine_tuned: false
    })
    
    // Load fine-tuned model
    await api.post('/chat/load-model', {
      run_id: currentRunId,
      use_fine_tuned: true
    })
    
    modelsLoaded.value = true
    showSuccess('Models loaded successfully!')
  } catch (err: any) {
    console.error('Failed to load models:', err)
    showError('Failed to load models: ' + (err.response?.data?.detail || err.message))
  }
})

// Methods
const sendMessage = async () => {
  let message = userInput.value.trim()
  if (!message || !runId.value) return
  
  // Security: Client-side validation
  const validation = validateInput(message)
  if (!validation.isValid) {
    showError(validation.error || 'Invalid message')
    return
  }
  
  // Security: Sanitize input before sending
  message = sanitizeInput(message)

  // Add user message to appropriate history based on mode
  if (chatMode.value === 'both' || chatMode.value === 'base') {
    leftChatHistory.value.push({ role: 'user', content: message })
  }
  if (chatMode.value === 'both' || chatMode.value === 'finetuned') {
    rightChatHistory.value.push({ role: 'user', content: message })
  }
  
  userInput.value = ''
  
  // Send to selected models
  if (chatMode.value === 'both' || chatMode.value === 'base') {
    isLeftGenerating.value = true
    generateLeftResponse(message)
  }
  
  if (chatMode.value === 'both' || chatMode.value === 'finetuned') {
    isRightGenerating.value = true
    generateRightResponse(message)
  }
}

const generateLeftResponse = async (message: string) => {
  const startTime = Date.now()
  
  try {
    const response = await api.post(`/chat/generate?run_id=${runId.value}&use_fine_tuned=false`, {
      message: message,
      system_prompt: leftConfig.value.systemPrompt,
      max_tokens: leftConfig.value.maxTokens,
      temperature: leftConfig.value.temperature,
      top_p: leftConfig.value.topP
    })
    
    const endTime = Date.now()
    const responseTime = ((endTime - startTime) / 1000).toFixed(1)
    const data = response.data
    
    leftChatHistory.value.push({
      role: 'assistant',
      content: data.text,
      metrics: {
        responseTime: responseTime,
        tokens: data.tokens,
        speed: data.tokens_per_second.toFixed(0)
      }
    })
    
    // Update stats
    const newCount = leftStats.value.count + 1
    leftStats.value.avgResponseTime = ((leftStats.value.avgResponseTime * leftStats.value.count) + parseFloat(responseTime)) / newCount
    leftStats.value.avgTokens = ((leftStats.value.avgTokens * leftStats.value.count) + data.tokens) / newCount
    leftStats.value.avgSpeed = ((leftStats.value.avgSpeed * leftStats.value.count) + data.tokens_per_second) / newCount
    leftStats.value.count = newCount
    
  } catch (err: any) {
    console.error('Base model generation failed:', err)
    leftChatHistory.value.push({
      role: 'assistant',
      content: 'Error: Failed to generate response from base model.',
      metrics: { responseTime: '0', tokens: 0, speed: 0 }
    })
  } finally {
    isLeftGenerating.value = false
  }
}

const generateRightResponse = async (message: string) => {
  const startTime = Date.now()
  
  try {
    const response = await api.post(`/chat/generate?run_id=${runId.value}&use_fine_tuned=true`, {
      message: message,
      system_prompt: rightConfig.value.systemPrompt,
      max_tokens: rightConfig.value.maxTokens,
      temperature: rightConfig.value.temperature,
      top_p: rightConfig.value.topP
    })
    
    const endTime = Date.now()
    const responseTime = ((endTime - startTime) / 1000).toFixed(1)
    const data = response.data
    
    rightChatHistory.value.push({
      role: 'assistant',
      content: data.text,
      metrics: {
        responseTime: responseTime,
        tokens: data.tokens,
        speed: data.tokens_per_second.toFixed(0)
      }
    })
    
    // Update stats
    const newCount = rightStats.value.count + 1
    rightStats.value.avgResponseTime = ((rightStats.value.avgResponseTime * rightStats.value.count) + parseFloat(responseTime)) / newCount
    rightStats.value.avgTokens = ((rightStats.value.avgTokens * rightStats.value.count) + data.tokens) / newCount
    rightStats.value.avgSpeed = ((rightStats.value.avgSpeed * rightStats.value.count) + data.tokens_per_second) / newCount
    rightStats.value.count = newCount
    
  } catch (err: any) {
    console.error('Fine-tuned model generation failed:', err)
    rightChatHistory.value.push({
      role: 'assistant',
      content: 'Error: Failed to generate response from fine-tuned model.',
      metrics: { responseTime: '0', tokens: 0, speed: 0 }
    })
  } finally {
    isRightGenerating.value = false
  }
}

const clearLeftChat = () => {
  leftChatHistory.value = []
}

const clearRightChat = () => {
  rightChatHistory.value = []
}

const resetStats = () => {
  leftStats.value = { avgResponseTime: 0, avgTokens: 0, avgSpeed: 0, count: 0 }
  rightStats.value = { avgResponseTime: 0, avgTokens: 0, avgSpeed: 0, count: 0 }
}

const exportModel = () => {
  router.push({ name: 'models' })
}

const openEditModal = () => {
  editName.value = fineTunedModelName.value
  editDescription.value = store.completedRun?.description || ''
  editTags.value = store.completedRun?.tags || ''
  editNotes.value = store.completedRun?.notes || ''
  showEditModal.value = true
}

const saveModelChanges = async () => {
  if (!runId.value) return
  
  isSaving.value = true
  try {
    await api.patch(`/training/runs/${runId.value}`, {
      name: editName.value.trim(),
      description: editDescription.value.trim(),
      tags: editTags.value.trim(),
      notes: editNotes.value.trim()
    })
    
    // Update store
    if (store.completedRun) {
      store.completedRun.name = editName.value.trim()
      store.completedRun.description = editDescription.value.trim()
      store.completedRun.tags = editTags.value.trim()
      store.completedRun.notes = editNotes.value.trim()
    }
    
    showSuccess('Model metadata updated successfully')
    showEditModal.value = false
  } catch (err: any) {
    console.error('Failed to update model:', err)
    showError('Failed to update model: ' + (err.response?.data?.detail || err.message))
  } finally {
    isSaving.value = false
  }
}

// Security: Format message content safely
const formatMessageContent = (content: string, role: 'user' | 'assistant'): string => {
  return formatMessage(content, role)
}
</script>

<style scoped>
/* Security: Safe message formatting styles */
.formatted-message :deep(h1),
.formatted-message :deep(h2),
.formatted-message :deep(h3),
.formatted-message :deep(h4),
.formatted-message :deep(h5),
.formatted-message :deep(h6) {
  margin-top: 1rem;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: inherit;
}

.formatted-message :deep(p) {
  margin-bottom: 0.5rem;
}

.formatted-message :deep(ul),
.formatted-message :deep(ol) {
  margin-left: 1.5rem;
  margin-bottom: 0.5rem;
}

.formatted-message :deep(li) {
  margin-bottom: 0.25rem;
}

.formatted-message :deep(code) {
  background-color: rgba(0, 0, 0, 0.3);
  padding: 0.125rem 0.25rem;
  border-radius: 0.25rem;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.875em;
}

.formatted-message :deep(pre) {
  background-color: rgba(0, 0, 0, 0.3);
  padding: 1rem;
  border-radius: 0.5rem;
  overflow-x: auto;
  margin: 0.5rem 0;
}

.formatted-message :deep(pre code) {
  background-color: transparent;
  padding: 0;
  border-radius: 0;
}

.formatted-message :deep(blockquote) {
  border-left: 3px solid rgba(255, 255, 255, 0.3);
  padding-left: 1rem;
  margin: 0.5rem 0;
  color: rgba(255, 255, 255, 0.8);
}

.formatted-message :deep(a) {
  color: #60a5fa;
  text-decoration: underline;
}

.formatted-message :deep(a:hover) {
  color: #93c5fd;
}

.formatted-message :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 0.5rem 0;
}

.formatted-message :deep(th),
.formatted-message :deep(td) {
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 0.5rem;
  text-align: left;
}

.formatted-message :deep(th) {
  background-color: rgba(255, 255, 255, 0.1);
  font-weight: 600;
}

.formatted-message :deep(hr) {
  border: none;
  border-top: 1px solid rgba(255, 255, 255, 0.2);
  margin: 1rem 0;
}
</style>
