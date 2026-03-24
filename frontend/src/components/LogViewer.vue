<template>
  <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold text-white">Detailed Training Log</h2>
      <div class="flex items-center gap-2">
        <button
          v-if="hasLog"
          @click="downloadLog"
          class="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors flex items-center gap-1.5"
        >
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download CSV
        </button>
        <button
          v-if="hasLog"
          @click="showModal = true"
          class="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
        >
          View Log
        </button>
      </div>
    </div>

    <div v-if="loading" class="text-center py-8">
      <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto"></div>
      <p class="mt-2 text-sm text-slate-400">Loading detailed log...</p>
    </div>

    <div v-else-if="!hasLog" class="text-center py-8 text-slate-500">
      <svg class="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <p class="text-sm">Detailed log not available</p>
      <p class="text-xs text-slate-600 mt-1">Log is generated during training</p>
    </div>

    <div v-else class="space-y-4">
      <!-- Summary Stats -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div class="bg-slate-800 rounded-lg p-3">
          <p class="text-xs text-slate-400 mb-1">Total Entries</p>
          <p class="text-lg font-semibold text-white">{{ logData.total_entries }}</p>
        </div>
        <div class="bg-slate-800 rounded-lg p-3">
          <p class="text-xs text-slate-400 mb-1">Final Loss</p>
          <p class="text-lg font-semibold text-green-400">{{ finalLoss }}</p>
        </div>
        <div class="bg-slate-800 rounded-lg p-3">
          <p class="text-xs text-slate-400 mb-1">Avg Speed</p>
          <p class="text-lg font-semibold text-blue-400">{{ avgSpeed }} tok/s</p>
        </div>
        <div class="bg-slate-800 rounded-lg p-3">
          <p class="text-xs text-slate-400 mb-1">Peak Memory</p>
          <p class="text-lg font-semibold text-orange-400">{{ peakMemory }} MB</p>
        </div>
      </div>

      <!-- Learning Rate Visualization -->
      <div class="bg-slate-800 rounded-lg p-4">
        <div class="flex items-center justify-between mb-2">
          <p class="text-sm font-medium text-white">Learning Rate</p>
          <p class="text-sm text-slate-400">{{ currentLR }}</p>
        </div>
        <div class="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div 
            class="h-full bg-blue-500 rounded-full transition-all duration-500"
            :style="{ width: lrPercentage + '%' }"
          ></div>
        </div>
        <div class="flex justify-between mt-1 text-xs text-slate-500">
          <span>0</span>
          <span>{{ maxLR }}</span>
        </div>
      </div>
    </div>

    <!-- Detailed Log Modal -->
    <Teleport to="body">
      <div
        v-if="showModal"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
        @click="showModal = false"
      >
        <div
          class="bg-slate-900 rounded-xl border border-slate-700 w-full max-w-4xl max-h-[80vh] flex flex-col shadow-2xl"
          @click.stop
        >
          <!-- Modal Header -->
          <div class="flex items-center justify-between p-4 border-b border-slate-800">
            <div>
              <h3 class="text-lg font-semibold text-white">Detailed Training Log</h3>
              <p class="text-sm text-slate-400">{{ logData.total_entries }} entries • Step-by-step metrics</p>
            </div>
            <div class="flex items-center gap-2">
              <button
                @click="downloadLog"
                class="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors flex items-center gap-1.5"
              >
                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                CSV
              </button>
              <button
                @click="showModal = false"
                class="p-1.5 text-slate-400 hover:text-white transition-colors"
              >
                <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          <!-- Log Table -->
          <div class="flex-1 overflow-auto p-4">
            <table class="w-full text-sm">
              <thead class="sticky top-0 bg-slate-800">
                <tr>
                  <th class="text-left py-2 px-3 text-slate-400 font-medium">Step</th>
                  <th class="text-left py-2 px-3 text-slate-400 font-medium">Loss</th>
                  <th class="text-left py-2 px-3 text-slate-400 font-medium">Learning Rate</th>
                  <th class="text-left py-2 px-3 text-slate-400 font-medium">Tokens/s</th>
                  <th class="text-left py-2 px-3 text-slate-400 font-medium">Memory (MB)</th>
                  <th class="text-left py-2 px-3 text-slate-400 font-medium">Time</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-800">
                <tr
                  v-for="entry in logData.entries"
                  :key="entry.step"
                  class="hover:bg-slate-800/50 transition-colors"
                >
                  <td class="py-2 px-3 text-slate-300">{{ entry.step }}</td>
                  <td class="py-2 px-3 font-medium" :class="getLossColor(entry.loss)">{{ entry.loss.toFixed(4) }}</td>
                  <td class="py-2 px-3 text-blue-400">{{ formatLR(entry.learning_rate) }}</td>
                  <td class="py-2 px-3 text-slate-300">{{ entry.tokens_per_second.toFixed(1) }}</td>
                  <td class="py-2 px-3 text-slate-300">{{ entry.memory_mb.toFixed(0) }}</td>
                  <td class="py-2 px-3 text-slate-500 text-xs">{{ formatTime(entry.timestamp) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'

interface LogEntry {
  timestamp: string
  step: number
  loss: number
  learning_rate: number
  tokens_per_second: number
  it_per_second: number
  cpu_percent: number
  memory_mb: number
  peak_memory_mb: number
}

interface LogData {
  run_id: string
  total_entries: number
  entries: LogEntry[]
}

const props = defineProps<{
  runId: string
}>()

const loading = ref(false)
const hasLog = ref(false)
const logData = ref<LogData>({ run_id: '', total_entries: 0, entries: [] })
const showModal = ref(false)

const api = axios.create({
  baseURL: '/api'
})

// Computed stats
const finalLoss = computed(() => {
  if (!logData.value.entries.length) return 'N/A'
  return logData.value.entries[logData.value.entries.length - 1].loss.toFixed(3)
})

const avgSpeed = computed(() => {
  if (!logData.value.entries.length) return 'N/A'
  const avg = logData.value.entries.reduce((sum, e) => sum + e.tokens_per_second, 0) / logData.value.entries.length
  return avg.toFixed(0)
})

const peakMemory = computed(() => {
  if (!logData.value.entries.length) return 'N/A'
  const max = Math.max(...logData.value.entries.map(e => e.peak_memory_mb))
  return max.toFixed(0)
})

const currentLR = computed(() => {
  if (!logData.value.entries.length) return 'N/A'
  const lr = logData.value.entries[logData.value.entries.length - 1].learning_rate
  return formatLR(lr)
})

const maxLR = computed(() => {
  if (!logData.value.entries.length) return '0.00e+00'
  const max = Math.max(...logData.value.entries.map(e => e.learning_rate))
  return formatLR(max)
})

const lrPercentage = computed(() => {
  if (!logData.value.entries.length) return 0
  const current = logData.value.entries[logData.value.entries.length - 1].learning_rate
  const max = Math.max(...logData.value.entries.map(e => e.learning_rate))
  return max > 0 ? (current / max) * 100 : 0
})

// Load detailed log
const loadLog = async () => {
  loading.value = true
  try {
    const response = await api.get(`/training/runs/${props.runId}/logs/detailed`)
    logData.value = response.data
    hasLog.value = true
  } catch (err: any) {
    if (err.response?.status === 404) {
      hasLog.value = false
    } else {
      console.error('Failed to load detailed log:', err)
    }
  } finally {
    loading.value = false
  }
}

// Download log as CSV
const downloadLog = () => {
  window.open(`/api/training/runs/${props.runId}/logs/detailed?format=csv`, '_blank')
}

// Format helpers
const formatLR = (lr: number): string => {
  if (lr >= 0.001) return lr.toFixed(3)
  return lr.toExponential(2)
}

const formatTime = (timestamp: string): string => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const getLossColor = (loss: number): string => {
  const entries = logData.value.entries
  if (!entries.length) return 'text-slate-300'
  const min = Math.min(...entries.map(e => e.loss))
  const max = Math.max(...entries.map(e => e.loss))
  const range = max - min
  if (range === 0) return 'text-slate-300'
  
  const normalized = (loss - min) / range
  if (normalized < 0.2) return 'text-green-400'
  if (normalized < 0.5) return 'text-blue-400'
  if (normalized < 0.8) return 'text-yellow-400'
  return 'text-red-400'
}

onMounted(() => {
  loadLog()
})
</script>