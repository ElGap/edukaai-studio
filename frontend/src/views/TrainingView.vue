<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="border-b border-slate-800 pb-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-white">Step 3: Training Execution</h1>
          <p class="mt-2 text-slate-400">
            Training in progress. Monitor loss curves and resource usage in real-time.
          </p>
        </div>
        <div class="flex items-center gap-3">
          <button
            v-if="isRunning"
            @click="pauseTraining"
            class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg font-medium transition-colors"
          >
            Pause
          </button>
          <button
            v-else-if="isPaused"
            @click="resumeTraining"
            class="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors"
          >
            Resume
          </button>
          <button
            @click="stopTraining"
            class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors"
          >
            Stop
          </button>
        </div>
      </div>
    </div>

    <div v-if="!activeRun" class="text-center py-12">
      <p class="text-slate-400">No active training run</p>
      <router-link
        to="/configure"
        class="mt-4 inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
      >
        Start Training
      </router-link>
    </div>

    <div v-else class="grid gap-6 lg:grid-cols-3">
      <!-- Main Content - Charts -->
      <div class="lg:col-span-2 space-y-6">
        <!-- Loss Curves Chart -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-lg font-semibold text-white">Loss Curves</h2>
            <div class="flex items-center gap-4 text-sm">
              <div class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full bg-blue-500"></span>
                <span class="text-slate-400">Training Loss</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full bg-red-500"></span>
                <span class="text-slate-400">Validation Loss</span>
              </div>
            </div>
          </div>
          <div class="h-80 bg-slate-800/50 rounded-lg p-4">
            <!-- SVG Loss Chart -->
            <svg v-if="lossHistory.length > 0 || valLossHistory.length > 0" class="w-full h-full" viewBox="0 0 800 300" preserveAspectRatio="none">
              <!-- Grid lines -->
              <g class="text-slate-700">
                <line x1="0" y1="0" x2="800" y2="0" stroke="currentColor" stroke-width="0.5"/>
                <line x1="0" y1="75" x2="800" y2="75" stroke="currentColor" stroke-width="0.5"/>
                <line x1="0" y1="150" x2="800" y2="150" stroke="currentColor" stroke-width="0.5"/>
                <line x1="0" y1="225" x2="800" y2="225" stroke="currentColor" stroke-width="0.5"/>
                <line x1="0" y1="300" x2="800" y2="300" stroke="currentColor" stroke-width="0.5"/>
              </g>
              
              <!-- X-axis labels -->
              <text x="0" y="315" fill="currentColor" class="text-xs text-slate-500">0</text>
              <text x="400" y="315" fill="currentColor" class="text-xs text-slate-500" text-anchor="middle">{{ Math.round(totalSteps / 2) }}</text>
              <text x="800" y="315" fill="currentColor" class="text-xs text-slate-500" text-anchor="end">{{ totalSteps }}</text>
              
              <!-- Training Loss curve -->
              <path
                v-if="lossHistory.length > 1"
                :d="trainingLossPath"
                fill="none"
                stroke="#3b82f6"
                stroke-width="2"
              />
              
              <!-- Validation Loss curve -->
              <path
                v-if="valLossHistory.length > 1"
                :d="validationLossPath"
                fill="none"
                stroke="#ef4444"
                stroke-width="2"
                stroke-dasharray="5,5"
              />
              
              <!-- Training Loss points -->
              <g v-for="(point, index) in visibleTrainingPoints" :key="'train-'+index">
                <circle
                  :cx="point.x"
                  :cy="point.y"
                  r="3"
                  fill="#3b82f6"
                />
              </g>
              
              <!-- Validation Loss points -->
              <g v-for="(point, index) in visibleValidationPoints" :key="'val-'+index">
                <circle
                  :cx="point.x"
                  :cy="point.y"
                  r="3"
                  fill="#ef4444"
                />
              </g>
            </svg>
            <div v-else class="flex items-center justify-center h-full">
              <p class="text-slate-500">Waiting for training data...</p>
            </div>
          </div>
          
          <!-- Chart Stats -->
          <div class="mt-4 flex justify-between text-sm">
            <span class="text-slate-400">Current Train Loss: {{ currentTrainLoss?.toFixed(4) || 'N/A' }}</span>
            <span class="text-slate-400">Current Val Loss: {{ currentValLoss?.toFixed(4) || 'N/A' }}</span>
            <span class="text-slate-400">Best Loss: {{ bestLoss?.toFixed(4) || 'N/A' }}</span>
          </div>
        </div>

        <!-- Live Logs -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-lg font-semibold text-white">Live Logs</h2>
            <div class="flex items-center gap-2">
              <span class="relative flex h-3 w-3">
                <span
                  v-if="isRunning"
                  class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"
                ></span>
                <span
                  :class="['relative inline-flex rounded-full h-3 w-3', isRunning ? 'bg-green-500' : 'bg-slate-500']"
                ></span>
              </span>
              <span class="text-sm text-slate-400">{{ connectionStatus }}</span>
            </div>
          </div>
          <div 
            ref="logsContainer" 
            class="h-64 bg-slate-950 rounded-lg p-4 font-mono text-sm overflow-y-auto space-y-1 scroll-smooth"
            @scroll="handleLogScroll"
          >
            <div
              v-for="(log, index) in logs"
              :key="index"
              :class="[
                'text-slate-300',
                log.level === 'error' ? 'text-red-400' : 
                log.level === 'warning' ? 'text-yellow-400' : 
                log.level === 'success' ? 'text-green-400' : ''
              ]"
            >
              <span class="text-slate-500">{{ log.timestamp }}</span>
              <span class="mx-2">|</span>
              {{ log.message }}
            </div>
            <div v-if="logs.length === 0" class="text-slate-600">
              Waiting for training to start...
            </div>
          </div>
          <div class="mt-2 flex justify-between items-center">
            <span class="text-xs text-slate-500">{{ logs.length }} log entries</span>
            <button 
              v-if="!autoScrollLogs" 
              @click="enableAutoScroll"
              class="text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              Resume auto-scroll
            </button>
          </div>
        </div>
      </div>

      <!-- Right Panel - Metrics -->
      <div class="space-y-6">
        <!-- Progress -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 class="text-lg font-semibold text-white mb-4">Progress</h2>
          
          <div class="mb-4">
            <div class="flex justify-between text-sm mb-2">
              <span class="text-slate-400">Step {{ currentStep }} / {{ totalSteps }}</span>
              <span class="text-white">{{ Math.round((currentStep / totalSteps) * 100) }}%</span>
            </div>
            <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                class="h-full bg-blue-600 transition-all duration-500"
                :style="{ width: `${(currentStep / totalSteps) * 100}%` }"
              ></div>
            </div>
          </div>

          <div class="space-y-3 text-sm">
            <div class="flex justify-between">
              <span class="text-slate-400">Best Loss</span>
              <span class="text-green-400 font-medium">{{ bestLoss?.toFixed(4) || 'N/A' }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">At Step</span>
              <span class="text-white">{{ bestStep || 'N/A' }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Est. Remaining</span>
              <span class="text-white">{{ estimatedRemaining }}</span>
            </div>
          </div>
        </div>

        <!-- Resource Usage -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 class="text-lg font-semibold text-white mb-4">Resource Usage</h2>
          
          <div class="space-y-4">
            <!-- CPU -->
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="text-slate-400">CPU</span>
                <span class="text-white">{{ cpuUsage.toFixed(1) }}%</span>
              </div>
              <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  class="h-full bg-purple-500 transition-all duration-500"
                  :style="{ width: `${Math.min(cpuUsage, 100)}%` }"
                ></div>
              </div>
            </div>

            <!-- GPU Memory -->
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="text-slate-400">GPU Memory</span>
                <span class="text-white">{{ gpuMemoryUsed.toFixed(1) }} / {{ gpuMemoryTotal }} GB</span>
              </div>
              <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  class="h-full bg-yellow-500 transition-all duration-500"
                  :style="{ width: `${Math.min(gpuMemoryPercent, 100)}%` }"
                ></div>
              </div>
            </div>

            <!-- RAM -->
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="text-slate-400">RAM</span>
                <span class="text-white">{{ ramUsed.toFixed(1) }} / {{ ramTotal }} GB</span>
              </div>
              <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  class="h-full bg-blue-500 transition-all duration-500"
                  :style="{ width: `${Math.min(ramPercent, 100)}%` }"
                ></div>
              </div>
            </div>
          </div>
        </div>

        <!-- Performance -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 class="text-lg font-semibold text-white mb-4">Performance</h2>
          
          <div class="grid grid-cols-2 gap-4">
            <div class="bg-slate-800 rounded-lg p-3 text-center">
              <p class="text-2xl font-bold text-white">{{ stepsPerSecond.toFixed(2) }}</p>
              <p class="text-xs text-slate-400">Steps/sec</p>
            </div>
            <div class="bg-slate-800 rounded-lg p-3 text-center">
              <p class="text-2xl font-bold text-white">{{ tokensPerSecond.toFixed(0) }}</p>
              <p class="text-xs text-slate-400">Tokens/sec</p>
            </div>
          </div>
          
          <div class="mt-4 text-sm">
            <div class="flex justify-between">
              <span class="text-slate-400">Peak Memory</span>
              <span class="text-white">{{ peakMemoryGB.toFixed(2) }} GB</span>
            </div>
            <div class="flex justify-between mt-1">
              <span class="text-slate-400">Peak CPU</span>
              <span class="text-white">{{ peakCpuPercent.toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Actions -->
    <div class="flex items-center justify-between pt-6 border-t border-slate-800">
      <router-link
        to="/configure"
        class="px-4 py-2 text-slate-400 hover:text-white transition-colors"
      >
        ← Back to Configure
      </router-link>
    </div>
    <!-- Stop Training Confirmation Modal -->
    <Teleport to="body">
      <div
        v-if="showStopModal"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
        @click="showStopModal = false"
      >
        <div
          class="bg-slate-900 rounded-xl border border-slate-700 p-6 max-w-md w-full mx-4 shadow-2xl"
          @click.stop
        >
          <div class="flex items-center gap-3 mb-4">
            <div class="w-10 h-10 rounded-full bg-red-900/50 flex items-center justify-center">
              <svg class="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h3 class="text-lg font-semibold text-white">Stop Training?</h3>
          </div>
          
          <p class="text-slate-400 mb-6">
            Are you sure you want to stop the training? This action cannot be undone and all progress will be saved.
          </p>
          
          <div class="flex gap-3 justify-end">
            <button
              @click="showStopModal = false"
              class="px-4 py-2 text-slate-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              @click="confirmStopTraining"
              class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors"
            >
              Yes, Stop Training
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useTrainingStore } from '@/stores/training'
import axios from 'axios'

const router = useRouter()
const store = useTrainingStore()

// API instance
const api = axios.create({
  baseURL: '/api'
})

const logsContainer = ref<HTMLDivElement | null>(null)

// State
const isRunning = ref(true)
const isPaused = ref(false)
const currentStep = ref(0)
const totalSteps = computed(() => store.activeRun?.total_steps || 1000)
const bestLoss = ref<number | undefined>(undefined)
const bestStep = ref<number | undefined>(undefined)
const logs = ref<Array<{ timestamp: string; message: string; level: string }>>([])
const lossHistory = ref<Array<{ step: number; loss: number }>>([])
const valLossHistory = ref<Array<{ step: number; loss: number }>>([])
const autoScrollLogs = ref(true)

// Resource metrics
const cpuUsage = ref(0)
const peakCpuPercent = ref(0)
const gpuMemoryUsed = ref(0)
const gpuMemoryTotal = ref(16)
const gpuMemoryPercent = computed(() => (gpuMemoryUsed.value / gpuMemoryTotal.value) * 100)
const ramUsed = ref(0)
const ramTotal = ref(32)
const ramPercent = computed(() => (ramUsed.value / ramTotal.value) * 100)
const peakMemoryGB = ref(0)

// Performance metrics
const stepsPerSecond = ref(0)
const tokensPerSecond = ref(0)

// WebSocket
const ws = ref<WebSocket | null>(null)
const connectionStatus = ref('Connecting...')

// Getters
const activeRun = computed(() => store.activeRun)
const currentTrainLoss = computed(() => {
  if (lossHistory.value.length === 0) return null
  return lossHistory.value[lossHistory.value.length - 1].loss
})
const currentValLoss = computed(() => {
  if (valLossHistory.value.length === 0) return null
  return valLossHistory.value[valLossHistory.value.length - 1].loss
})

const estimatedRemaining = computed(() => {
  if (stepsPerSecond.value === 0) return 'Calculating...'
  const remaining = totalSteps.value - currentStep.value
  const seconds = remaining / stepsPerSecond.value
  const minutes = Math.ceil(seconds / 60)
  
  if (minutes < 60) {
    return `${minutes}m`
  }
  const hours = Math.floor(minutes / 60)
  const mins = minutes % 60
  return `${hours}h ${mins}m`
})

// Loss curve paths with proper X-axis scaling to totalSteps
const trainingLossPath = computed(() => {
  if (lossHistory.value.length < 2) return ''
  
  const allLosses = [
    ...lossHistory.value.map(p => p.loss),
    ...(valLossHistory.value.length > 0 ? valLossHistory.value.map(p => p.loss) : [])
  ]
  const minLoss = Math.min(...allLosses)
  const maxLoss = Math.max(...allLosses)
  const lossRange = maxLoss - minLoss || 1
  
  // Scale X from 0 to totalSteps (800 pixels)
  const points = lossHistory.value.map(point => ({
    x: (point.step / totalSteps.value) * 800,
    y: 300 - ((point.loss - minLoss) / lossRange) * 280 - 10
  }))
  
  if (points.length < 2) return ''
  
  let path = `M ${points[0].x} ${points[0].y}`
  for (let i = 1; i < points.length; i++) {
    path += ` L ${points[i].x} ${points[i].y}`
  }
  return path
})

const validationLossPath = computed(() => {
  if (valLossHistory.value.length < 2) return ''
  
  const allLosses = [
    ...lossHistory.value.map(p => p.loss),
    ...valLossHistory.value.map(p => p.loss)
  ]
  const minLoss = Math.min(...allLosses)
  const maxLoss = Math.max(...allLosses)
  const lossRange = maxLoss - minLoss || 1
  
  // Scale X from 0 to totalSteps (800 pixels)
  const points = valLossHistory.value.map(point => ({
    x: (point.step / totalSteps.value) * 800,
    y: 300 - ((point.loss - minLoss) / lossRange) * 280 - 10
  }))
  
  if (points.length < 2) return ''
  
  let path = `M ${points[0].x} ${points[0].y}`
  for (let i = 1; i < points.length; i++) {
    path += ` L ${points[i].x} ${points[i].y}`
  }
  return path
})

const visibleTrainingPoints = computed(() => {
  if (lossHistory.value.length === 0) return []
  
  const allLosses = [
    ...lossHistory.value.map(p => p.loss),
    ...(valLossHistory.value.length > 0 ? valLossHistory.value.map(p => p.loss) : [])
  ]
  const minLoss = Math.min(...allLosses)
  const maxLoss = Math.max(...allLosses)
  const lossRange = maxLoss - minLoss || 1
  
  return lossHistory.value.map(point => ({
    x: (point.step / totalSteps.value) * 800,
    y: 300 - ((point.loss - minLoss) / lossRange) * 280 - 10
  }))
})

const visibleValidationPoints = computed(() => {
  if (valLossHistory.value.length === 0) return []
  
  const allLosses = [
    ...lossHistory.value.map(p => p.loss),
    ...valLossHistory.value.map(p => p.loss)
  ]
  const minLoss = Math.min(...allLosses)
  const maxLoss = Math.max(...allLosses)
  const lossRange = maxLoss - minLoss || 1
  
  return valLossHistory.value.map(point => ({
    x: (point.step / totalSteps.value) * 800,
    y: 300 - ((point.loss - minLoss) / lossRange) * 280 - 10
  }))
})

// Methods
const addLog = (message: string, level: string = 'info') => {
  console.log('addLog called:', message, 'Current logs count:', logs.value.length)
  const timestamp = new Date().toLocaleTimeString()
  logs.value.push({ timestamp, message, level })
  console.log('Log added. New count:', logs.value.length)
  
  // Keep only last 100 logs
  if (logs.value.length > 100) {
    logs.value = logs.value.slice(-100)
  }
  
  // Auto-scroll to bottom
  if (autoScrollLogs.value) {
    nextTick(() => {
      if (logsContainer.value) {
        logsContainer.value.scrollTop = logsContainer.value.scrollHeight
      }
    })
  }
}

const handleLogScroll = () => {
  if (!logsContainer.value) return
  
  const { scrollTop, scrollHeight, clientHeight } = logsContainer.value
  const isAtBottom = scrollHeight - scrollTop - clientHeight < 10
  autoScrollLogs.value = isAtBottom
}

const enableAutoScroll = () => {
  autoScrollLogs.value = true
  nextTick(() => {
    if (logsContainer.value) {
      logsContainer.value.scrollTop = logsContainer.value.scrollHeight
    }
  })
}

const pauseTraining = () => {
  isRunning.value = false
  isPaused.value = true
  addLog('Training paused', 'warning')
  
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({ action: 'pause' }))
  }
}

const resumeTraining = () => {
  isRunning.value = true
  isPaused.value = false
  addLog('Training resumed', 'success')
  
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({ action: 'resume' }))
  }
}

const stopTraining = () => {
  showStopModal.value = true
}

const showStopModal = ref(false)

const confirmStopTraining = () => {
  showStopModal.value = false
  isRunning.value = false
  addLog('Training stopped by user', 'warning')
  
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({ action: 'stop' }))
  }
  
  if (ws.value) {
    ws.value.close()
  }
  
  // Don't redirect to summary when stopped - just stay on training page
  // User can manually navigate if they want to see partial results
}

const connectWebSocket = () => {
  if (!activeRun.value) return
  
  const runId = activeRun.value.id
  const wsUrl = `ws://localhost:8000/api/ws/training/runs/${runId}`
  
  connectionStatus.value = 'Connecting...'
  addLog('Connecting to training stream...', 'info')
  
  ws.value = new WebSocket(wsUrl)
  
  ws.value.onopen = () => {
    connectionStatus.value = 'Connected'
    addLog('Connected to training stream', 'success')
    
    // Stop polling when WebSocket is connected
    stopStatusPolling()
    
    // Clear previous training metrics when starting new training
    store.clearTrainingMetrics()
    
    const pingInterval = setInterval(() => {
      if (ws.value && ws.value.readyState === WebSocket.OPEN) {
        ws.value.send(JSON.stringify({ action: 'ping' }))
      } else {
        clearInterval(pingInterval)
      }
    }, 30000)
  }
  
  ws.value.onmessage = (event) => {
    console.log('WebSocket message received:', event.data)
    const data = JSON.parse(event.data)
    console.log('Parsed data:', data)
    
    if (data.type === 'training_update') {
      const stats = data.data
      
      // Save metric to store for Summary page curves
      store.addTrainingMetric({
        step: stats.current_step,
        train_loss: stats.current_loss,
        eval_loss: stats.validation_loss,
        learning_rate: stats.learning_rate,
        tokens_per_second: stats.tokens_per_second,
        cpu_percent: stats.peak_cpu_percent,
        gpu_memory_used_mb: stats.peak_memory_mb
      })
      
      // Update progress
      currentStep.value = stats.current_step
      if (stats.best_loss !== undefined && stats.best_loss !== null) {
        bestLoss.value = stats.best_loss
        bestStep.value = stats.best_step
      }
      
      // Add loss to history - include step 0 and always include last point
      if (stats.current_loss !== undefined && stats.current_loss !== null) {
        lossHistory.value.push({
          step: stats.current_step,
          loss: stats.current_loss
        })
        
        // Add validation loss if available (only when validation runs, typically every 100 steps)
        if (stats.validation_loss !== undefined && stats.validation_loss !== null) {
          valLossHistory.value.push({
            step: stats.current_step,
            loss: stats.validation_loss
          })
        }
      }
        
        // Log training progress periodically (every 10 steps)
        if (stats.current_step % 10 === 0 || stats.current_step === 1) {
          const progress = ((stats.current_step / totalSteps.value) * 100).toFixed(1)
          const speed = stats.tokens_per_second ? `${stats.tokens_per_second.toFixed(0)} tok/s` : ''
          const mem = stats.peak_memory_mb ? `${(stats.peak_memory_mb/1024).toFixed(1)}GB` : ''
          addLog(
            `Step ${stats.current_step}/${totalSteps.value} (${progress}%) | ` +
            `Loss: ${stats.current_loss.toFixed(4)} | ` +
            `Best: ${stats.best_loss?.toFixed(4) || 'N/A'} @ Step ${stats.best_step || 0} | ` +
            `${speed} ${mem}`, 
            'info'
          )
        }
        
        // Update resource metrics from backend stats
        if (stats.peak_memory_mb !== undefined) {
        peakMemoryGB.value = stats.peak_memory_mb / 1024
        gpuMemoryUsed.value = stats.peak_memory_mb / 1024
      }
      if (stats.peak_cpu_percent !== undefined) {
        peakCpuPercent.value = stats.peak_cpu_percent
        cpuUsage.value = stats.peak_cpu_percent
      }
      
      // Update performance metrics
      if (stats.tokens_per_second !== undefined) {
        tokensPerSecond.value = stats.tokens_per_second
      }
      if (stats.it_per_second !== undefined) {
        stepsPerSecond.value = stats.it_per_second
      }
      
      // Update RAM usage from resources if available
      if (stats.resources && stats.resources.memory_mb) {
        ramUsed.value = stats.resources.memory_mb / 1024
      }
      
      // Update status
      console.log('Status update:', stats.status, stats.status_message)
      if (stats.status === 'downloading') {
        // Show detailed download progress message if available
        console.log('Download status detected, message:', stats.status_message)
        if (stats.status_message) {
          addLog(`${stats.status_message}`, 'info')
        } else {
          addLog('Downloading model from HuggingFace...', 'info')
        }
      } else if (stats.status === 'loading_model') {
        // Model loading status shown in app logs and training log only
        // Not showing in live chat to avoid clutter
      } else if (stats.status === 'model_loaded') {
        addLog('Model loaded successfully', 'success')
      } else if (stats.status === 'running') {
        isRunning.value = true
        isPaused.value = false
      } else if (stats.status === 'paused') {
        isRunning.value = false
        isPaused.value = true
        addLog('Training paused', 'warning')
      } else if (stats.status === 'stopped') {
        isRunning.value = false
        addLog('Training stopped by user', 'warning')
        // Don't redirect to summary when stopped - just stay on training page
      } else if (stats.status === 'completed') {
        isRunning.value = false
        const duration = stats.start_time && stats.end_time ? 
          ((new Date(stats.end_time).getTime() - new Date(stats.start_time).getTime()) / 60000).toFixed(1) : 'N/A'
        addLog(
          `Training completed! Best Loss: ${stats.best_loss?.toFixed(4) || 'N/A'} ` +
          `at Step ${stats.best_step || 0} | Duration: ${duration} min`, 
          'success'
        )
        
        store.setCompletedRun({
          id: activeRun.value?.id || 'mock-run',
          name: activeRun.value?.name || 'Mock Run',
          status: 'completed',
          current_step: currentStep.value,
          total_steps: totalSteps.value,
          best_loss: bestLoss.value,
          best_step: bestStep.value,
          validation_loss: currentValLoss.value || undefined,
          created_at: activeRun.value?.created_at || new Date().toISOString(),
          completed_at: new Date().toISOString(),
          base_model: activeRun.value?.base_model || { id: '1', huggingface_id: '', name: 'Mock Model', architecture: '', parameter_count: 0, context_length: 0 }
        })
        
        router.push({ name: 'summary' })
      } else if (stats.status === 'failed') {
        isRunning.value = false
        const errorMsg = stats.error_message || 'Unknown error'
        addLog(`Training failed: ${errorMsg}`, 'error')
      }
    }
    
    if (data.type === 'pong') {
      // Connection is alive
    }
    
    if (data.type === 'error') {
      addLog(`Error: ${data.message}`, 'error')
    }
  }
  
  ws.value.onerror = (error) => {
    connectionStatus.value = 'Error'
    addLog('WebSocket error occurred', 'error')
    console.error('WebSocket error:', error)
  }
  
  ws.value.onclose = async () => {
    connectionStatus.value = 'Disconnected'
    addLog('Disconnected from training stream', 'warning')
    
    // Start polling when WebSocket disconnects
    startStatusPolling()
    
    // Check if training completed while WebSocket was closing
    if (activeRun.value) {
      try {
        const response = await api.get(`/training/runs/${activeRun.value.id}`)
        const runData = response.data
        
        if (runData.status === 'completed') {
          addLog('Training completed! Redirecting to summary...', 'success')
          store.setCompletedRun(runData)
          router.push({ name: 'summary' })
          stopStatusPolling()
        }
      } catch (err) {
        console.error('Failed to check run status after WebSocket close:', err)
      }
    }
  }
}

// Status polling interval (for when WebSocket is disconnected)
let statusPollInterval: number | null = null

const startStatusPolling = () => {
  if (statusPollInterval) return
  
  statusPollInterval = window.setInterval(async () => {
    if (!activeRun.value || connectionStatus.value === 'Connected') return
    
    try {
      const response = await api.get(`/training/runs/${activeRun.value.id}`)
      const runData = response.data
      
      // Update UI with latest status
      if (runData.status !== activeRun.value.status) {
        console.log(`Status changed via polling: ${activeRun.value.status} -> ${runData.status}`)
        
        // Update the run in store
        const runIndex = store.trainingRuns.findIndex(r => r.id === runData.id)
        if (runIndex !== -1) {
          store.trainingRuns[runIndex] = runData
        }
        
        // Handle completion
        if (runData.status === 'completed') {
          addLog('Training completed! Redirecting to summary...', 'success')
          store.setCompletedRun(runData)
          router.push({ name: 'summary' })
          stopStatusPolling()
        }
      }
      
      // Update metrics if available
      if (runData.current_step !== currentStep.value) {
        currentStep.value = runData.current_step
      }
      if (runData.best_loss !== bestLoss.value) {
        bestLoss.value = runData.best_loss
      }
    } catch (err) {
      console.error('Failed to poll status:', err)
    }
  }, 5000) // Poll every 5 seconds
}

const stopStatusPolling = () => {
  if (statusPollInterval) {
    clearInterval(statusPollInterval)
    statusPollInterval = null
  }
}

// Lifecycle
onMounted(async () => {
  console.log('TrainingView mounted, checking active run...')
  console.log('activeRunId from store:', store.activeRunId)
  console.log('trainingRuns count:', store.trainingRuns.length)
  
  // Wait a bit for store to be ready
  await new Promise(resolve => setTimeout(resolve, 100))
  
  console.log('After delay - activeRun:', activeRun.value)
  
  if (!activeRun.value) {
    console.log('No active run found after delay, redirecting to configure')
    router.push({ name: 'configure' })
    return
  }
  
  // Check if training is already completed - redirect to summary
  if (activeRun.value.status === 'completed') {
    console.error('Training has already completed. View results in Summary.')
    router.push({ name: 'summary' })
    return
  }
  
  // Initialize total steps from active run
  if (activeRun.value.total_steps) {
    currentStep.value = activeRun.value.current_step || 0
  }
  
  // Show initial status
  addLog('Initializing training environment...', 'info')
  
  // Connect to WebSocket for real-time updates
  connectWebSocket()
  
  // Only start training if status is pending
  if (activeRun.value.status === 'pending') {
    addLog('Starting training run...', 'info')
    try {
      await api.post(`/training/runs/${activeRun.value.id}/start`)
      addLog('Training started successfully!', 'success')
    } catch (error: any) {
      console.error('Failed to start training:', error)
      addLog(`Failed to start training: ${error.response?.data?.detail || error.message}`, 'error')
    }
  } else if (activeRun.value.status === 'running') {
    addLog('Training is in progress...', 'info')
  } else if (activeRun.value.status === 'paused') {
    addLog('Training is paused. Click Resume to continue.', 'warning')
  } else if (activeRun.value.status === 'failed') {
    addLog('Previous training attempt failed. You can retry with different settings.', 'warning')
  } else if (activeRun.value.status === 'stopped') {
    addLog('Training was stopped. Starting new training session...', 'info')
    try {
      await api.post(`/training/runs/${activeRun.value.id}/start`)
      addLog('Training restarted successfully!', 'success')
    } catch (error: any) {
      console.error('Failed to restart training:', error)
      addLog(`Failed to restart: ${error.response?.data?.detail || error.message}`, 'error')
    }
  }
})

onUnmounted(() => {
  if (ws.value) {
    ws.value.close()
  }
  stopStatusPolling()
})
</script>
