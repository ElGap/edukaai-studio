<template>
  <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
    <h2 class="text-lg font-semibold text-white mb-4">Export Model</h2>
    <p class="text-sm text-slate-400 mb-4">
      Export your fine-tuned model in different formats for various use cases.
    </p>

    <div class="space-y-3">
      <!-- Adapter Export -->
      <div class="flex items-center justify-between p-4 bg-slate-800 rounded-lg">
        <div class="flex items-start gap-3">
          <div class="w-10 h-10 rounded-lg bg-blue-900/50 flex items-center justify-center flex-shrink-0">
            <svg class="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div>
            <div class="flex items-center gap-2">
              <h3 class="font-medium text-white">LoRA Adapter</h3>
              <span v-if="exportStatus.adapter.available" class="px-2 py-0.5 bg-green-900/50 text-green-400 text-xs rounded-full">
                ✓ Ready
              </span>
            </div>
            <p class="text-sm text-slate-400 mt-1">
              Lightweight LoRA weights only (~10-50MB). Requires base model.
            </p>
            <p v-if="exportStatus.adapter.size_mb" class="text-xs text-slate-500 mt-1">
              Size: {{ exportStatus.adapter.size_mb }} MB
            </p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <button
            v-if="exportStatus.adapter.available"
            @click="downloadExport('adapter')"
            class="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
          >
            Download
          </button>
          <button
            v-else
            @click="exportModel('adapter')"
            :disabled="isExporting"
            class="px-3 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-500 text-white text-sm rounded-lg transition-colors"
          >
            <span v-if="exportingFormat === 'adapter'" class="flex items-center gap-2">
              <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Exporting...
            </span>
            <span v-else>Export</span>
          </button>
        </div>
      </div>

      <!-- Fused Export -->
      <div class="flex items-center justify-between p-4 bg-slate-800 rounded-lg">
        <div class="flex items-start gap-3">
          <div class="w-10 h-10 rounded-lg bg-purple-900/50 flex items-center justify-center flex-shrink-0">
            <svg class="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <div>
            <div class="flex items-center gap-2">
              <h3 class="font-medium text-white">Fused Model</h3>
              <span v-if="exportStatus.fused.available" class="px-2 py-0.5 bg-green-900/50 text-green-400 text-xs rounded-full">
                ✓ Ready
              </span>
            </div>
            <p class="text-sm text-slate-400 mt-1">
              Complete model with LoRA merged (~2-4GB). Standalone model.
            </p>
            <p v-if="exportStatus.fused.size_mb" class="text-xs text-slate-500 mt-1">
              Size: {{ exportStatus.fused.size_mb }} MB
            </p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <button
            v-if="exportStatus.fused.available"
            @click="downloadExport('fused')"
            class="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
          >
            Download
          </button>
          <button
            v-else
            @click="exportModel('fused')"
            :disabled="isExporting"
            class="px-3 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-500 text-white text-sm rounded-lg transition-colors"
          >
            <span v-if="exportingFormat === 'fused'" class="flex items-center gap-2">
              <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Exporting...
            </span>
            <span v-else>Export</span>
          </button>
        </div>
      </div>

      <!-- GGUF Export -->
      <div class="flex items-center justify-between p-4 bg-slate-800 rounded-lg">
        <div class="flex items-start gap-3">
          <div class="w-10 h-10 rounded-lg bg-orange-900/50 flex items-center justify-center flex-shrink-0">
            <svg class="w-5 h-5 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
          </div>
          <div>
            <div class="flex items-center gap-2">
              <h3 class="font-medium text-white">GGUF Format</h3>
              <span v-if="exportStatus.gguf.available" class="px-2 py-0.5 bg-green-900/50 text-green-400 text-xs rounded-full">
                ✓ Ready
              </span>
              <span class="px-2 py-0.5 bg-yellow-900/50 text-yellow-400 text-xs rounded-full">
                Beta
              </span>
            </div>
            <p class="text-sm text-slate-400 mt-1">
              Optimized for llama.cpp/ollama (~1-2GB). CPU inference ready.
            </p>
            <p v-if="exportStatus.gguf.size_mb" class="text-xs text-slate-500 mt-1">
              Size: {{ exportStatus.gguf.size_mb }} MB
            </p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <button
            v-if="exportStatus.gguf.available"
            @click="downloadExport('gguf')"
            class="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
          >
            Download
          </button>
          <button
            v-else
            @click="exportModel('gguf')"
            :disabled="isExporting"
            class="px-3 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-500 text-white text-sm rounded-lg transition-colors"
          >
            <span v-if="exportingFormat === 'gguf'" class="flex items-center gap-2">
              <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Exporting...
            </span>
            <span v-else>Export</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Error Message -->
    <div v-if="error" class="mt-4 p-3 bg-red-900/30 border border-red-800 rounded-lg">
      <p class="text-sm text-red-400">{{ error }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'
import type { TrainingRun } from '@/stores/training'

const props = defineProps<{
  model: TrainingRun
}>()

const emit = defineEmits<{
  exportComplete: [format: string]
}>()

const api = axios.create({
  baseURL: '/api'
})

// Export status
const exportStatus = ref({
  adapter: { available: false, path: null as string | null, size_mb: null as number | null, exported_at: null as string | null },
  fused: { available: false, path: null as string | null, size_mb: null as number | null, exported_at: null as string | null },
  gguf: { available: false, path: null as string | null, size_mb: null as number | null, exported_at: null as string | null }
})

const isExporting = ref(false)
const exportingFormat = ref<string | null>(null)
const error = ref<string | null>(null)

// Load export status
const loadExportStatus = async () => {
  try {
    const response = await api.get(`/training/runs/${props.model.id}/exports/status`)
    exportStatus.value = response.data
  } catch (err: any) {
    console.error('Failed to load export status:', err)
  }
}

// Export model
const exportModel = async (format: 'adapter' | 'fused' | 'gguf') => {
  isExporting.value = true
  exportingFormat.value = format
  error.value = null

  try {
    await api.post(`/training/runs/${props.model.id}/exports`, {
      format
    })
    
    // Update status
    await loadExportStatus()
    
    // Emit event
    emit('exportComplete', format)
  } catch (err: any) {
    console.error('Export failed:', err)
    error.value = err.response?.data?.detail || 'Export failed'
  } finally {
    isExporting.value = false
    exportingFormat.value = null
  }
}

// Download export
const downloadExport = (format: 'adapter' | 'fused' | 'gguf') => {
  window.open(`/api/training/runs/${props.model.id}/exports/${format}/download`, '_blank')
}

// Load status on mount
onMounted(() => {
  loadExportStatus()
})
</script>