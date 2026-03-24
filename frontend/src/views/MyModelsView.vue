<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="border-b border-slate-800 pb-6">
      <h1 class="text-2xl font-bold text-white">My Models</h1>
      <p class="mt-2 text-slate-400">
        Manage all your fine-tuned models
      </p>
    </div>

    <!-- Filters -->
    <div class="flex flex-wrap gap-4">
      <input
        v-model="searchQuery"
        type="text"
        placeholder="Search models..."
        class="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
      />
      <select
        v-model="filterBaseModel"
        class="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
      >
        <option value="">All Base Models</option>
        <option value="qwen">Qwen</option>
        <option value="llama">Llama</option>
        <option value="phi">Phi</option>
      </select>
      <select
        v-model="sortBy"
        class="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
      >
        <option value="date">Sort by Date</option>
        <option value="loss">Sort by Loss</option>
        <option value="name">Sort by Name</option>
        <option value="tags">Sort by Tags</option>
      </select>
    </div>

    <!-- Bulk Actions Bar -->
    <div v-if="selectedModels.length > 0" class="flex items-center gap-4 px-4 py-3 bg-blue-900/30 border border-blue-800 rounded-lg">
      <span class="text-sm text-blue-300">
        {{ selectedModels.length }} model{{ selectedModels.length !== 1 ? 's' : '' }} selected
      </span>
      <div class="flex gap-2 ml-auto">
        <button
          @click="bulkDelete"
          class="px-3 py-1.5 text-sm bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
          :disabled="isBulkDeleting"
        >
          <span v-if="isBulkDeleting">Deleting...</span>
          <span v-else>Delete Selected</span>
        </button>
        <button
          @click="clearSelection"
          class="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
        >
          Clear
        </button>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="text-center py-12">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
      <p class="mt-4 text-slate-400">Loading models...</p>
    </div>

    <!-- Models Table -->
    <div v-else class="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
      <div class="overflow-x-auto max-h-[70vh] overflow-y-auto">
        <table class="w-full text-left">
          <!-- Sticky Header -->
          <thead class="bg-slate-800 sticky top-0 z-10">
            <tr>
              <th class="px-4 py-3 w-8">
                <input
                  type="checkbox"
                  :checked="isAllSelected"
                  @change="toggleSelectAll"
                  class="rounded border-slate-600 bg-slate-700 text-blue-600 focus:ring-blue-500"
                />
              </th>
              <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider w-1/4">Model Name</th>
              <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Status</th>
              <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Base Model</th>
              <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">Best Loss</th>
              <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-center">Progress</th>
              <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Exports</th>
              <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Tags</th>
              <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Created</th>
              <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">Actions</th>
            </tr>
          </thead>
          
          <!-- Table Body -->
          <tbody class="divide-y divide-slate-800">
            <tr
              v-for="model in filteredModels"
              :key="model.id"
              class="hover:bg-slate-800/50 transition-colors"
              :class="{ 'bg-blue-900/20': selectedModels.includes(model.id) }"
            >
              <!-- Checkbox -->
              <td class="px-4 py-3">
                <input
                  type="checkbox"
                  :value="model.id"
                  v-model="selectedModels"
                  class="rounded border-slate-600 bg-slate-700 text-blue-600 focus:ring-blue-500"
                />
              </td>
              
              <!-- Model Name -->
              <td class="px-4 py-3">
                <div>
                  <h3 class="font-medium text-white text-sm">{{ model.name }}</h3>
                  <p class="text-xs text-slate-500 mt-0.5">ID: {{ model.id.slice(0, 8) }}...</p>
                </div>
              </td>
              
              <!-- Status -->
              <td class="px-4 py-3">
                <span
                  :class="[
                    'inline-flex px-2 py-1 rounded-full text-xs font-medium',
                    model.status === 'completed' ? 'bg-green-600 text-white' : 
                    model.status === 'failed' ? 'bg-red-600 text-white' : 
                    model.status === 'running' ? 'bg-blue-600 text-white' :
                    'bg-yellow-600 text-white'
                  ]"
                >
                  {{ model.status }}
                </span>
              </td>
              
              <!-- Base Model -->
              <td class="px-4 py-3">
                <span class="text-sm text-slate-300">{{ model.base_model?.name || 'Unknown' }}</span>
              </td>
              
              <!-- Best Loss -->
              <td class="px-4 py-3 text-right">
                <span 
                  class="text-sm font-medium"
                  :class="model.best_loss ? 'text-green-400' : 'text-slate-500'"
                >
                  {{ model.best_loss?.toFixed(4) || 'N/A' }}
                </span>
              </td>
              
              <!-- Progress -->
              <td class="px-4 py-3">
                <div class="flex items-center gap-2">
                  <div class="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      class="h-full bg-blue-500 rounded-full"
                      :style="{ width: `${(model.current_step / model.total_steps) * 100}%` }"
                    ></div>
                  </div>
                  <span class="text-xs text-slate-400 whitespace-nowrap">
                    {{ model.current_step }}/{{ model.total_steps }}
                  </span>
                </div>
              </td>
              
              <!-- Exports -->
              <td class="px-4 py-3">
                <div v-if="model.status === 'completed'" class="flex gap-1">
                  <span 
                    v-if="model.adapter_exported" 
                    class="px-1.5 py-0.5 bg-blue-900/50 text-blue-400 text-[10px] rounded"
                    title="LoRA Adapter"
                  >
                    LORA
                  </span>
                  <span 
                    v-if="model.fused_exported" 
                    class="px-1.5 py-0.5 bg-purple-900/50 text-purple-400 text-[10px] rounded"
                    title="Fused"
                  >
                    FUSED
                  </span>
                  <span 
                    v-if="model.gguf_exported" 
                    class="px-1.5 py-0.5 bg-orange-900/50 text-orange-400 text-[10px] rounded"
                    title="GGUF"
                  >
                    GGUF
                  </span>
                  <span 
                    v-if="!model.adapter_exported && !model.fused_exported && !model.gguf_exported" 
                    class="text-xs text-slate-500"
                  >
                    None
                  </span>
                </div>
                <span v-else class="text-xs text-slate-500">-</span>
              </td>
              
              <!-- Tags -->
              <td class="px-4 py-3">
                <div v-if="model.tags" class="flex flex-wrap gap-1">
                  <span 
                    v-for="tag in model.tags.split(',').map(t => t.trim()).filter(t => t)" 
                    :key="tag"
                    class="px-2 py-0.5 bg-slate-700/50 text-slate-300 text-[10px] rounded-full"
                  >
                    {{ tag }}
                  </span>
                </div>
                <span v-else class="text-xs text-slate-500">-</span>
              </td>
              
              <!-- Created -->
              <td class="px-4 py-3">
                <div class="flex flex-col">
                  <span class="text-xs text-slate-300">
                    {{ new Date(model.created_at).toLocaleDateString() }}
                  </span>
                  <span class="text-[10px] text-slate-500">
                    {{ new Date(model.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }}
                  </span>
                </div>
              </td>
              
              <!-- Actions -->
              <td class="px-4 py-3 text-right">
                <div class="flex justify-end gap-1">
                  <button
                    v-if="model.status === 'completed'"
                    @click="viewSummary(model)"
                    class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
                    title="View Summary"
                  >
                    Summary
                  </button>
                  <button
                    v-if="model.status === 'completed'"
                    @click="chatModel(model)"
                    class="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                    title="Chat Model"
                  >
                    Chat
                  </button>
                  <button
                    v-if="model.status === 'completed'"
                    @click="showExportPanel(model)"
                    class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
                    title="Export"
                  >
                    Export
                  </button>
                  <button
                    @click="editModel(model)"
                    class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
                    title="Edit"
                  >
                    Edit
                  </button>
                  <button
                    @click="deleteModel(model)"
                    class="px-2 py-1 text-xs text-red-400 hover:text-red-300 hover:bg-red-900/20 rounded transition-colors"
                    title="Delete"
                  >
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      
      <!-- Table Footer -->
      <div class="px-4 py-3 bg-slate-800/50 border-t border-slate-800 text-xs text-slate-500">
        Showing {{ filteredModels.length }} model{{ filteredModels.length !== 1 ? 's' : '' }}
        <span v-if="models.length !== filteredModels.length"> (filtered from {{ models.length }})</span>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="!loading && filteredModels.length === 0" class="text-center py-12">
      <p class="text-slate-400">No models found</p>
      <router-link
        to="/configure"
        class="mt-4 inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
      >
        Create Your First Model
      </router-link>
    </div>

    <!-- Delete Confirmation Modal -->
    <Teleport to="body">
      <div
        v-if="showDeleteModal"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
        @click="showDeleteModal = false"
      >
        <div
          class="bg-slate-900 rounded-xl border border-slate-700 p-6 max-w-md w-full mx-4 shadow-2xl"
          @click.stop
        >
          <div class="flex items-center gap-3 mb-4">
            <div class="w-10 h-10 rounded-full bg-red-900/50 flex items-center justify-center">
              <svg class="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </div>
            <h3 class="text-lg font-semibold text-white">Delete Model?</h3>
          </div>
          
          <p class="text-slate-400 mb-2">
            Are you sure you want to delete <strong class="text-white">{{ modelToDelete?.name }}</strong>?
          </p>
          <p class="text-sm text-slate-500 mb-6">
            This will permanently remove the model and all associated training data. This action cannot be undone.
          </p>
          
          <div class="flex gap-3 justify-end">
            <button
              @click="showDeleteModal = false"
              class="px-4 py-2 text-slate-400 hover:text-white transition-colors"
              :disabled="isDeleting"
            >
              Cancel
            </button>
            <button
              @click="confirmDelete"
              class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
              :disabled="isDeleting"
            >
              <svg v-if="isDeleting" class="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {{ isDeleting ? 'Deleting...' : 'Yes, Delete' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Bulk Delete Confirmation Modal -->
    <Teleport to="body">
      <div
        v-if="showBulkDeleteModal"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
        @click="showBulkDeleteModal = false"
      >
        <div
          class="bg-slate-900 rounded-xl border border-slate-700 p-6 max-w-md w-full mx-4 shadow-2xl"
          @click.stop
        >
          <div class="flex items-center gap-3 mb-4">
            <div class="w-10 h-10 rounded-full bg-red-900/50 flex items-center justify-center">
              <svg class="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </div>
            <h3 class="text-lg font-semibold text-white">Delete {{ selectedModels.length }} Models?</h3>
          </div>
          
          <p class="text-slate-400 mb-2">
            Are you sure you want to delete <strong class="text-white">{{ selectedModels.length }} model{{ selectedModels.length !== 1 ? 's' : '' }}</strong>?
          </p>
          <p class="text-sm text-slate-500 mb-6">
            This will permanently remove all selected models and their associated training data. This action cannot be undone.
          </p>
          
          <div class="flex gap-3 justify-end">
            <button
              @click="showBulkDeleteModal = false"
              class="px-4 py-2 text-slate-400 hover:text-white transition-colors"
              :disabled="isBulkDeleting"
            >
              Cancel
            </button>
            <button
              @click="executeBulkDelete"
              class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
              :disabled="isBulkDeleting"
            >
              <svg v-if="isBulkDeleting" class="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {{ isBulkDeleting ? 'Deleting...' : 'Yes, Delete All' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Export Modal -->
    <Teleport to="body">
      <div
        v-if="showExportModal"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
        @click="showExportModal = false"
      >
        <div
          class="bg-slate-900 rounded-xl border border-slate-700 p-6 max-w-lg w-full mx-4 shadow-2xl"
          @click.stop
        >
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-white">Export Model</h3>
            <button 
              @click="showExportModal = false"
              class="text-slate-400 hover:text-white"
            >
              <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <ExportPanel v-if="modelToExport" :model="modelToExport" @export-complete="onExportComplete" />
        </div>
      </div>
    </Teleport>

    <!-- Edit Modal -->
    <Teleport to="body">
      <div
        v-if="showEditModal"
        class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
        @click="showEditModal = false"
      >
        <div
          class="bg-slate-900 rounded-xl border border-slate-700 p-6 max-w-lg w-full mx-4 shadow-2xl"
          @click.stop
        >
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-white">Edit Model</h3>
            <button 
              @click="showEditModal = false"
              class="text-slate-400 hover:text-white"
            >
              <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-1">Model Name</label>
              <input
                v-model="editName"
                type="text"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="Enter model name"
              />
            </div>
            
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-1">Description</label>
              <textarea
                v-model="editDescription"
                rows="3"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none resize-none"
                placeholder="What did you train this model for? (e.g., customer support bot, code assistant, medical QA)"
              ></textarea>
              <p class="text-xs text-slate-500 mt-1">{{ editDescription.length }}/5000 characters</p>
            </div>
            
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-1">Tags</label>
              <input
                v-model="editTags"
                type="text"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="Comma-separated tags (e.g., customer-support, gpt-style, v1)"
              />
              <p class="text-xs text-slate-500 mt-1">Separate tags with commas</p>
            </div>
            
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-1">
                Notes & Findings
                <span class="text-xs text-slate-400 ml-1">(optional)</span>
              </label>
              <textarea
                v-model="editNotes"
                rows="5"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none resize-none"
                placeholder="Document your fine-tuning observations: what worked, what didn't, insights gained, lessons learned..."
              ></textarea>
              <p class="text-xs text-slate-500 mt-1">{{ editNotes.length }}/10000 characters</p>
            </div>
          </div>
          
          <div class="flex gap-3 justify-end mt-6">
            <button
              @click="showEditModal = false"
              class="px-4 py-2 text-slate-300 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              @click="saveModelEdits"
              :disabled="isSaving || !editName.trim()"
              class="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            >
              <svg v-if="isSaving" class="animate-spin h-4 w-4 text-white inline mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {{ isSaving ? 'Saving...' : 'Save Changes' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useTrainingStore } from '@/stores/training'
import type { TrainingRun } from '@/stores/training'
import axios from 'axios'
import { useToast } from '@/composables/useToast'
import ExportPanel from '@/components/ExportPanel.vue'

const router = useRouter()
const store = useTrainingStore()
const { success: showSuccess, error: showError } = useToast()

const api = axios.create({
  baseURL: '/api'
})

const searchQuery = ref('')
const filterBaseModel = ref('')
const sortBy = ref('date')
const models = ref<TrainingRun[]>([])
const loading = ref(true)

// Delete modal state
const showDeleteModal = ref(false)
const modelToDelete = ref<TrainingRun | null>(null)
const isDeleting = ref(false)

// Bulk delete modal state
const showBulkDeleteModal = ref(false)

// Export modal state
const showExportModal = ref(false)
const modelToExport = ref<TrainingRun | null>(null)

// Edit modal state
const showEditModal = ref(false)
const modelToEdit = ref<TrainingRun | null>(null)
const editName = ref('')
const editDescription = ref('')
const editTags = ref('')
const editNotes = ref('')
const isSaving = ref(false)

// Bulk selection state
const selectedModels = ref<string[]>([])
const isBulkDeleting = ref(false)

const isAllSelected = computed(() => {
  return filteredModels.value.length > 0 && selectedModels.value.length === filteredModels.value.length
})

const toggleSelectAll = () => {
  if (isAllSelected.value) {
    selectedModels.value = []
  } else {
    selectedModels.value = filteredModels.value.map(m => m.id)
  }
}

const clearSelection = () => {
  selectedModels.value = []
}

const bulkDelete = () => {
  if (selectedModels.value.length === 0) return
  showBulkDeleteModal.value = true
}

const executeBulkDelete = async () => {
  isBulkDeleting.value = true
  let deleted = 0
  let failed = 0
  
  for (const modelId of selectedModels.value) {
    try {
      await api.delete(`/training/runs/${modelId}`)
      deleted++
    } catch (err) {
      console.error(`Failed to delete ${modelId}:`, err)
      failed++
    }
  }
  
  // Refresh models list
  await loadModels()
  selectedModels.value = []
  showBulkDeleteModal.value = false
  
  if (failed === 0) {
    showSuccess(`Deleted ${deleted} models`)
  } else {
    showError(`Deleted ${deleted} models, ${failed} failed`)
  }
  
  isBulkDeleting.value = false
}

// Load real models from API
const loadModels = async () => {
  try {
    loading.value = true
    const response = await api.get('/training/runs')
    models.value = response.data
    showSuccess(`Loaded ${models.value.length} models`)
  } catch (err: any) {
    console.error('Failed to load models:', err)
    showError('Failed to load models: ' + (err.response?.data?.detail || err.message))
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadModels()
})

const filteredModels = computed(() => {
  let result = models.value

  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(m => m.name.toLowerCase().includes(query))
  }

  if (filterBaseModel.value) {
    result = result.filter(m => m.base_model?.name?.toLowerCase().includes(filterBaseModel.value))
  }

  if (sortBy.value === 'date') {
    result = result.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
  } else if (sortBy.value === 'loss') {
    result = result.sort((a, b) => (a.best_loss || 999) - (b.best_loss || 999))
  } else if (sortBy.value === 'name') {
    result = result.sort((a, b) => a.name.localeCompare(b.name))
  } else if (sortBy.value === 'tags') {
    // Sort by first tag alphabetically, models without tags at the end
    result = result.sort((a, b) => {
      const tagA = (a.tags || '').split(',')[0]?.trim().toLowerCase() || 'zzz'
      const tagB = (b.tags || '').split(',')[0]?.trim().toLowerCase() || 'zzz'
      return tagA.localeCompare(tagB)
    })
  }

  return result
})

const viewSummary = (model: TrainingRun) => {
  store.setCompletedRun(model)
  router.push({ name: 'summary' })
}

const chatModel = (model: TrainingRun) => {
  store.setCompletedRun(model)
  router.push({ name: 'chat' })
}

const showExportPanel = (model: TrainingRun) => {
  modelToExport.value = model
  showExportModal.value = true
}

const onExportComplete = async (format: string) => {
  showSuccess(`Export completed: ${format}`)
  // Refresh models to update export flags
  await loadModels()
}

const editModel = (model: TrainingRun) => {
  modelToEdit.value = model
  editName.value = model.name
  editDescription.value = model.description || ''
  editTags.value = model.tags || ''
  editNotes.value = model.notes || ''
  showEditModal.value = true
}

const saveModelEdits = async () => {
  if (!modelToEdit.value) return
  
  isSaving.value = true
  
  try {
    const response = await api.patch(`/training/runs/${modelToEdit.value.id}`, {
      name: editName.value,
      description: editDescription.value,
      tags: editTags.value,
      notes: editNotes.value
    })
    
    // Update local model
    const index = models.value.findIndex(m => m.id === modelToEdit.value?.id)
    if (index !== -1) {
      models.value[index] = response.data
    }
    
    showSuccess(`Updated model: ${editName.value}`)
    showEditModal.value = false
    modelToEdit.value = null
  } catch (err: any) {
    console.error('Update failed:', err)
    showError('Failed to update model: ' + (err.response?.data?.detail || err.message))
  } finally {
    isSaving.value = false
  }
}

const deleteModel = (model: TrainingRun) => {
  modelToDelete.value = model
  showDeleteModal.value = true
}

const confirmDelete = async () => {
  if (!modelToDelete.value) return
  
  isDeleting.value = true
  
  try {
    await api.delete(`/training/runs/${modelToDelete.value.id}`)
    models.value = models.value.filter(m => m.id !== modelToDelete.value?.id)
    showSuccess(`Deleted ${modelToDelete.value.name}`)
    showDeleteModal.value = false
    modelToDelete.value = null
  } catch (err: any) {
    console.error('Delete failed:', err)
    showError('Delete failed: ' + (err.response?.data?.detail || err.message))
  } finally {
    isDeleting.value = false
  }
}
</script>
