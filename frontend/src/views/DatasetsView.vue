<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="border-b border-slate-800 pb-6">
      <h1 class="text-2xl font-bold text-white">Step 1: Upload Dataset</h1>
      <p class="mt-2 text-slate-400">
        Upload your training data in JSONL format (Alpaca, ShareGPT, or custom)
      </p>
    </div>
    
    <!-- Upload Section -->
    <div class="grid gap-6 lg:grid-cols-2">
      <!-- Main Upload -->
      <div class="space-y-6">
        <div
          @dragover.prevent
          @drop.prevent="handleDrop"
          class="border-2 border-dashed border-slate-700 rounded-xl p-8 text-center hover:border-blue-500 transition-colors cursor-pointer"
          @click="triggerFileInput"
        >
          <input
            ref="fileInput"
            type="file"
            accept=".jsonl,.json"
            class="hidden"
            @change="handleFileChange"
          />
          
          <svg class="mx-auto h-12 w-12 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          
          <p class="mt-4 text-lg font-medium text-white">
            Drop your dataset here
          </p>
          <p class="mt-2 text-sm text-slate-400">
            or click to browse
          </p>
          <p class="mt-1 text-xs text-slate-500">
            JSONL format, up to 10,000 samples
          </p>
        </div>
        
        <!-- Validation Dataset Option -->
        <div class="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <div class="flex items-center justify-between">
            <div>
              <h3 class="font-medium text-white">Optional: Upload Validation Dataset</h3>
              <p class="text-sm text-slate-400 mt-1">
                Separate file for validation metrics
              </p>
            </div>
            <button
              v-if="!validationFile"
              @click="triggerValidationFileInput"
              class="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-medium text-white transition-colors"
            >
              Upload
            </button>
            <div v-else class="flex items-center gap-3">
              <span class="text-sm text-green-400">✓ {{ validationFile.name }}</span>
              <button
                @click="validationFile = null"
                class="text-slate-500 hover:text-red-400"
              >
                Remove
              </button>
            </div>
          </div>
          
          <input
            ref="validationFileInput"
            type="file"
            accept=".jsonl,.json"
            class="hidden"
            @change="handleValidationFileChange"
          />
        </div>
        
        <!-- Auto-split Option -->
        <div v-if="!validationFile" class="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <h3 class="font-medium text-white mb-3">Auto-split Validation</h3>
          <p class="text-sm text-slate-400 mb-4">
            Automatically split your dataset into training and validation sets
          </p>
          
          <div class="flex items-center gap-3 mb-4">
            <input
              v-model="useAutoSplit"
              type="checkbox"
              id="auto-split"
              class="w-4 h-4 rounded border-slate-600 bg-slate-800 text-blue-600 focus:ring-blue-500"
            />
            <label for="auto-split" class="text-white cursor-pointer">
              Enable auto-split
            </label>
          </div>
          
          <div v-if="useAutoSplit" class="mt-4">
            <label class="block text-sm font-medium text-slate-300 mb-2">
              Validation Split ({{ validationSplitPercent }}%)
            </label>
            <div class="flex gap-2">
              <button
                v-for="percent in [5, 10, 15]"
                :key="percent"
                @click="validationSplitPercent = percent"
                :class="[
                  'px-3 py-2 rounded-lg text-sm font-medium transition-colors flex-1',
                  validationSplitPercent === percent
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                ]"
              >
                {{ percent }}%
              </button>
            </div>
            <p class="text-xs text-slate-500 mt-2">
              <span v-if="validationSplitPercent === 5" class="text-green-400">
                95% training, 5% validation - More training data
              </span>
              <span v-else-if="validationSplitPercent === 10" class="text-blue-400">
                90% training, 10% validation - Balanced (standard)
              </span>
              <span v-else-if="validationSplitPercent === 15" class="text-orange-400">
                85% training, 15% validation - More validation data
              </span>
            </p>
          </div>
        </div>
      </div>
      
      <!-- Preview Section -->
      <div v-if="uploadedDataset" class="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
        <div class="p-4 border-b border-slate-800 bg-slate-800/50">
          <div class="flex items-center justify-between">
            <h3 class="font-medium text-white">Dataset Preview</h3>
            <span
              :class="[
                'px-2 py-1 rounded-full text-xs font-medium',
                uploadedDataset.format === 'chat' ? 'bg-purple-900/50 text-purple-300' :
                uploadedDataset.format === 'completion' ? 'bg-blue-900/50 text-blue-300' :
                'bg-slate-700 text-slate-300'
              ]"
            >
              {{ uploadedDataset.format }}
            </span>
          </div>
        </div>
        
        <div class="p-4 space-y-4">
          <!-- Stats -->
          <div class="grid grid-cols-2 gap-4">
            <div class="bg-slate-800 rounded-lg p-3">
              <p class="text-xs text-slate-500">Total Samples</p>
              <p class="text-lg font-semibold text-white">{{ uploadedDataset.num_samples }}</p>
            </div>
            <div class="bg-slate-800 rounded-lg p-3">
              <p class="text-xs text-slate-500">Valid / Invalid</p>
              <p class="text-lg font-semibold text-white">
                {{ uploadedDataset.validation_report?.valid_samples || uploadedDataset.num_samples }} / {{ uploadedDataset.validation_report?.invalid_samples || 0 }}
              </p>
            </div>
          </div>
          
          <!-- Preview Samples -->
          <div v-if="uploadedDataset.preview_samples?.length > 0" class="space-y-3">
            <p class="text-xs font-medium text-slate-500 uppercase">
              Preview: {{ uploadedDataset.preview_samples.length }} samples (First 10 + Random 10)
            </p>
            
            <div
              v-for="(sample, index) in uploadedDataset.preview_samples.slice(0, 6)"
              :key="index"
              class="bg-slate-800 rounded-lg p-3 text-sm"
            >
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs text-slate-500">
                  {{ index < 10 ? `Sample ${index + 1} (First)` : `Sample ${index + 1} (Random)` }}
                </span>
              </div>
              <pre class="text-slate-300 overflow-x-auto whitespace-pre-wrap text-xs">{{ JSON.stringify(sample, null, 2) }}</pre>
            </div>
            
            <div v-if="(uploadedDataset.preview_samples?.length || 0) > 6" class="text-center">
              <button 
                @click="showAllSamples = !showAllSamples"
                class="text-xs text-blue-400 hover:text-blue-300 transition-colors"
              >
                {{ showAllSamples ? 'Show Less' : `+ ${(uploadedDataset.preview_samples?.length || 0) - 6} more samples` }}
              </button>
            </div>
            
            <!-- Show all samples when expanded -->
            <div v-if="showAllSamples && uploadedDataset.preview_samples" class="space-y-3">
              <div
                v-for="(sample, index) in (uploadedDataset.preview_samples || []).slice(6)"
                :key="index + 6"
                class="bg-slate-800 rounded-lg p-3 text-sm"
              >
                <div class="flex items-center justify-between mb-1">
                  <span class="text-xs text-slate-500">
                    {{ (index + 6) < 10 ? `Sample ${index + 7} (First)` : `Sample ${index + 7} (Random)` }}
                  </span>
                </div>
                <pre class="text-slate-300 overflow-x-auto whitespace-pre-wrap text-xs">{{ JSON.stringify(sample, null, 2) }}</pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Actions -->
    <div class="flex items-center justify-between pt-6 border-t border-slate-800">
      <div class="text-sm text-slate-400">
        <span v-if="uploadedDataset" class="text-green-400">
          ✓ Dataset ready for training
        </span>
        <span v-else>
          Upload a dataset to continue
        </span>
      </div>
      
      <button
        @click="proceedToConfigure"
        :disabled="!uploadedDataset"
        :class="[
          'px-6 py-3 rounded-lg font-medium transition-all',
          uploadedDataset
            ? 'bg-blue-600 hover:bg-blue-700 text-white'
            : 'bg-slate-700 text-slate-500 cursor-not-allowed'
        ]"
      >
        Configure Training →
      </button>
    </div>
    
    <!-- Dataset List -->
    <div v-if="datasets.length > 0" class="pt-8">
      <h2 class="text-lg font-semibold text-white mb-4">Your Datasets</h2>
      <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <div
          v-for="dataset in datasets"
          :key="dataset.id"
          @click="selectDataset(dataset)"
          :class="[
            'p-4 rounded-xl border cursor-pointer transition-all',
            selectedDataset?.id === dataset.id
              ? 'border-blue-500 bg-blue-900/20'
              : 'border-slate-800 bg-slate-900 hover:border-slate-700'
          ]"
        >
          <div class="flex items-start justify-between">
            <div class="flex-1 min-w-0">
              <h3 class="font-medium text-white truncate">{{ dataset.name }}</h3>
              <p class="text-sm text-slate-400 mt-1">{{ dataset.num_samples }} samples</p>
              <!-- Validation Set Badge -->
              <div v-if="dataset.has_validation_set" class="flex items-center gap-2 mt-2">
                <span class="px-2 py-0.5 bg-green-900/50 text-green-400 text-xs rounded-full flex items-center gap-1">
                  <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  {{ dataset.validation_samples }} validation
                </span>
              </div>
            </div>
            <span
              :class="[
                'px-2 py-1 rounded-full text-xs font-medium',
                dataset.format === 'chat' ? 'bg-purple-900/50 text-purple-300' :
                dataset.format === 'completion' ? 'bg-blue-900/50 text-blue-300' :
                'bg-slate-700 text-slate-300'
              ]"
            >
              {{ dataset.format }}
            </span>
          </div>
          <p class="text-xs text-slate-500 mt-3">
            {{ new Date(dataset.created_at).toLocaleDateString() }}
          </p>
          
          <!-- Action Buttons -->
          <div class="flex items-center gap-2 mt-3 pt-3 border-t border-slate-800">
            <button
              v-if="!dataset.has_validation_set"
              @click.stop="triggerValidationUpload(dataset)"
              class="flex-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded font-medium transition-colors"
            >
              Add Validation
            </button>
            <button
              v-else
              @click.stop="viewValidationSet(dataset)"
              class="flex-1 px-3 py-1.5 bg-green-900/30 hover:bg-green-900/50 text-green-400 text-xs rounded font-medium transition-colors"
            >
              View Validation
            </button>
            <button
              @click.stop="configureTraining(dataset)"
              class="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded font-medium transition-colors"
            >
              Train
            </button>
            <button
              @click.stop="confirmDelete(dataset)"
              class="p-1.5 text-slate-400 hover:text-red-400 transition-colors"
              title="Delete dataset"
            >
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
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
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h3 class="text-lg font-semibold text-white">Delete Dataset?</h3>
          </div>
          
          <p class="text-slate-400 mb-2">
            Are you sure you want to delete <strong class="text-white">{{ datasetToDelete?.name }}</strong>?
          </p>
          <p class="text-sm text-slate-500 mb-6">
            This will remove {{ datasetToDelete?.num_samples }} samples. This action cannot be undone.
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
              @click="deleteDataset"
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
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useTrainingStore } from '@/stores/training'
import type { Dataset } from '@/stores/training'
import axios from 'axios'
import { useToast } from '@/composables/useToast'

const router = useRouter()
const store = useTrainingStore()
const { error: showError, success: showSuccess } = useToast()

// API client
const api = axios.create({
  baseURL: '/api'
})

const fileInput = ref<HTMLInputElement | null>(null)
const validationFileInput = ref<HTMLInputElement | null>(null)
const uploadedDataset = ref<Dataset | null>(null)
const validationFile = ref<File | null>(null)
const useAutoSplit = ref(true)
const validationSplitPercent = ref(10)  // 5, 10, or 15%
const datasetForValidation = ref<Dataset | null>(null)
const showAllSamples = ref(false)  // Toggle for showing all preview samples

const datasets = ref<Dataset[]>([])
const selectedDataset = ref<Dataset | null>(null)

const JSON = window.JSON

onMounted(async () => {
  // Load existing datasets from API
  await loadDatasets()
})

const loadDatasets = async () => {
  try {
    const response = await api.get('/datasets')
    datasets.value = response.data
    // Also populate the store
    store.setDatasets(response.data)
  } catch (err: any) {
    console.error('Failed to load datasets:', err)
    showError('Failed to load existing datasets')
  }
}

const triggerFileInput = () => {
  fileInput.value?.click()
}

const triggerValidationFileInput = () => {
  validationFileInput.value?.click()
}

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    uploadDataset(file)
  }
}

const handleValidationFileChange = async (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  
  // If triggered from dataset card (datasetForValidation set), upload it
  if (file && datasetForValidation.value) {
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('name', `${datasetForValidation.value.name} (Validation)`)
      
      const response = await api.post(`/datasets/${datasetForValidation.value.id}/validation`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      
      // Update the dataset in the list
      const datasetIndex = datasets.value.findIndex(d => d.id === datasetForValidation.value?.id)
      if (datasetIndex > -1) {
        datasets.value[datasetIndex].has_validation_set = true
        datasets.value[datasetIndex].validation_set_id = response.data.id
        datasets.value[datasetIndex].validation_samples = response.data.num_samples
      }
      
      showSuccess(`Validation set uploaded with ${response.data.num_samples} samples`)
      await loadDatasets()
    } catch (err: any) {
      console.error('Failed to upload validation set:', err)
      const errorMessage = err.response?.data?.detail || 'Failed to upload validation set'
      showError(errorMessage)
    } finally {
      datasetForValidation.value = null
      if (validationFileInput.value) {
        validationFileInput.value.value = ''
      }
    }
  } else if (file) {
    // Just store for the main upload flow
    validationFile.value = file
  }
}

const handleDrop = (event: DragEvent) => {
  const file = event.dataTransfer?.files[0]
  if (file && (file.name.endsWith('.jsonl') || file.name.endsWith('.json'))) {
    uploadDataset(file)
  }
}

const uploadDataset = async (file: File) => {
  try {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('name', file.name.replace(/\.jsonl?$/, ''))
    
    const response = await api.post('/datasets', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    const dataset: Dataset = {
      id: response.data.id,
      name: response.data.name,
      description: response.data.description,
      format: response.data.format,
      num_samples: response.data.num_samples,
      preview_samples: response.data.preview_samples || [],
      created_at: response.data.created_at,
      validation_report: response.data.validation_report
    }
    
    uploadedDataset.value = dataset
    selectedDataset.value = dataset
    store.setSelectedDataset(dataset.id)
    showAllSamples.value = false  // Reset sample view toggle
    
    // Add to datasets list (local and store)
    datasets.value.unshift(dataset)
    store.addDataset(dataset)
    
    showSuccess(`Dataset "${dataset.name}" uploaded successfully with ${dataset.num_samples} samples`)
  } catch (err: any) {
    console.error('Failed to upload dataset:', err)
    const errorMessage = err.response?.data?.detail || 'Failed to upload dataset. Please check the file format.'
    showError(errorMessage)
  } finally {
    // Reset file input to allow re-uploading the same file
    if (fileInput.value) {
      fileInput.value.value = ''
    }
  }
}

const selectDataset = (dataset: Dataset) => {
  selectedDataset.value = dataset
  uploadedDataset.value = dataset
  store.setSelectedDataset(dataset.id)
}

const proceedToConfigure = () => {
  if (uploadedDataset.value) {
    // Save validation split preference to store
    store.setValidationSplitPercent(validationSplitPercent.value)
    router.push({ name: 'configure' })
  }
}

// Delete dataset functionality
const showDeleteModal = ref(false)
const datasetToDelete = ref<Dataset | null>(null)
const isDeleting = ref(false)

const confirmDelete = (dataset: Dataset) => {
  datasetToDelete.value = dataset
  showDeleteModal.value = true
}

const deleteDataset = async () => {
  if (!datasetToDelete.value) return
  
  isDeleting.value = true
  
  try {
    await api.delete(`/datasets/${datasetToDelete.value.id}`)
    
    // Remove from local list
    const index = datasets.value.findIndex(d => d.id === datasetToDelete.value?.id)
    if (index > -1) {
      datasets.value.splice(index, 1)
    }
    
    // If the deleted dataset was selected, clear it
    if (selectedDataset.value?.id === datasetToDelete.value.id) {
      selectedDataset.value = null
      uploadedDataset.value = null
      store.setSelectedDataset('')
    }
    
    showSuccess(`Dataset "${datasetToDelete.value.name}" deleted successfully`)
    showDeleteModal.value = false
    datasetToDelete.value = null
  } catch (err: any) {
    console.error('Failed to delete dataset:', err)
    const errorMessage = err.response?.data?.detail || 'Failed to delete dataset'
    showError(errorMessage)
  } finally {
    isDeleting.value = false
  }
}

// Configure training with specific dataset
const configureTraining = (dataset: Dataset) => {
  // Set this dataset as selected
  store.setSelectedDataset(dataset.id)
  
  // Also update local state
  selectedDataset.value = dataset
  uploadedDataset.value = dataset
  
  // Navigate to configure page
  router.push({ name: 'configure' })
}

// Validation set handling
const triggerValidationUpload = (dataset: Dataset) => {
  datasetForValidation.value = dataset
  validationFileInput.value?.click()
}

const viewValidationSet = (dataset: Dataset) => {
  if (dataset.validation_set_id) {
    showSuccess(`Validation set: ${dataset.validation_samples} samples`)
  }
}

</script>
