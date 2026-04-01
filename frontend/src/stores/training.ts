import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

const api = axios.create({
  baseURL: '/api'
})

export interface Dataset {
  id: string
  name: string
  description?: string
  format: string
  num_samples: number
  preview_samples: any[]
  created_at: string
  validation_report?: {
    total_samples: number
    valid_samples: number
    invalid_samples: number
    errors: any[]
    format_detected: string
    preview?: {
      first_5: any[]
      random_5: any[]
    }
  }
  has_validation_set?: boolean
  validation_set_id?: string
  validation_samples?: number
}

export interface BaseModel {
  id: string
  huggingface_id: string
  name: string
  architecture: string
  parameter_count: number
  context_length: number
  mlx_config?: {
    is_curated?: boolean
    is_custom?: boolean
    [key: string]: any
  }
}

export interface TrainingPreset {
  id: string
  name: string
  description?: string
  is_default?: boolean
  steps: number
  learning_rate: number
  lora_rank: number
  lora_alpha: number
  lora_dropout?: number
  batch_size: number
  warmup_steps?: number
  gradient_accumulation_steps?: number
  early_stopping_patience?: number
  gradient_checkpointing?: boolean
  num_lora_layers?: number
  prompt_masking?: boolean
}

export interface TrainingConfig {
  steps: number
  learning_rate: number
  lora_rank: number
  lora_alpha: number
  lora_dropout: number
  batch_size: number
  max_seq_length: number
  warmup_steps: number
  gradient_accumulation_steps: number
  early_stopping_patience: number
  gradient_checkpointing: boolean
  num_lora_layers: number
  prompt_masking: boolean
  validation_split_percent: number  // 5, 10, or 15
  dataset?: {
    id: string
    name: string
    num_samples: number
    anonymization_report?: {
      total_samples: number
      samples_with_pii: number
      total_replacements: number
      types_found: Record<string, number>
      fields_affected: string[]
    }
  }
}

export interface TrainingRun {
  id: string
  name: string
  description?: string
  tags?: string
  notes?: string  // User notes/thoughts about fine-tuning
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped'
  current_step: number
  total_steps: number
  best_loss?: number
  best_step?: number
  validation_loss?: number
  error_message?: string
  created_at: string
  completed_at?: string
  base_model: BaseModel
  adapter_exported?: boolean
  fused_exported?: boolean
  gguf_exported?: boolean
  training_config?: TrainingConfig
}

export interface TrainingMetric {
  step: number
  timestamp?: string
  train_loss?: number
  eval_loss?: number
  learning_rate?: number
  gradient_norm?: number
  cpu_percent?: number
  memory_percent?: number
  gpu_memory_used_mb?: number
  samples_per_second?: number
  tokens_per_second?: number
  elapsed_seconds?: number
}

export const useTrainingStore = defineStore('training', () => {
  // State - all UI state is memory-only (no localStorage for security)
  const datasets = ref<Dataset[]>([])
  const baseModels = ref<BaseModel[]>([])
  const presets = ref<TrainingPreset[]>([])
  const trainingRuns = ref<TrainingRun[]>([])
  
  const selectedDatasetId = ref<string | null>(null)
  const selectedValidationDatasetId = ref<string | null>(null)
  const activeRunId = ref<string | null>(null)
  const completedRun = ref<TrainingRun | null>(null)
  const validationSplitPercent = ref<number>(10)
  const trainingMetrics = ref<TrainingMetric[]>([])
  
  // No localStorage persistence for active run - security focused
  const setActiveRun = (id: string | null) => {
    activeRunId.value = id
  }
  
  // Training runs fetched from database - no localStorage needed
  const addTrainingRun = (run: TrainingRun) => {
    trainingRuns.value.push(run)
  }
  
  const setTrainingRuns = (runs: TrainingRun[]) => {
    trainingRuns.value = runs
  }
  
  // Getters
  const selectedDataset = computed(() => 
    datasets.value.find(d => d.id === selectedDatasetId.value)
  )
  
  const activeRun = computed(() => 
    trainingRuns.value.find(r => r.id === activeRunId.value)
  )
  
  // Actions
  const setSelectedDataset = (id: string) => {
    selectedDatasetId.value = id
  }
  
  const addDataset = (dataset: Dataset) => {
    // Add to beginning of array if not already present
    const existingIndex = datasets.value.findIndex(d => d.id === dataset.id)
    if (existingIndex === -1) {
      datasets.value.unshift(dataset)
    }
  }
  
  const setDatasets = (newDatasets: Dataset[]) => {
    datasets.value = newDatasets
  }
  
  const setBaseModels = (models: BaseModel[]) => {
    baseModels.value = models
  }
  
  const setPresets = (newPresets: TrainingPreset[]) => {
    presets.value = newPresets
  }
  
  const setCompletedRun = (run: TrainingRun) => {
    completedRun.value = run
  }
  
  const fetchCompletedRun = async (runId: string) => {
    try {
      const response = await api.get(`/training/runs/${runId}`)
      completedRun.value = response.data
      return response.data
    } catch (error) {
      console.error('Failed to fetch training run:', error)
      return null
    }
  }
  
  const setValidationSplitPercent = (percent: number) => {
    validationSplitPercent.value = percent
  }
  
  const addTrainingMetric = (metric: TrainingMetric) => {
    trainingMetrics.value.push(metric)
  }
  
  const setTrainingMetrics = (metrics: TrainingMetric[]) => {
    trainingMetrics.value = metrics
  }
  
  const clearTrainingMetrics = () => {
    trainingMetrics.value = []
  }
  
  const clearTrainingState = () => {
    selectedDatasetId.value = null
    selectedValidationDatasetId.value = null
    activeRunId.value = null
    completedRun.value = null
    validationSplitPercent.value = 10
    trainingMetrics.value = []
    // No localStorage to clear - all state is memory-only
  }
  
  return {
    // State
    datasets,
    baseModels,
    presets,
    trainingRuns,
    selectedDatasetId,
    selectedValidationDatasetId,
    activeRunId,
    completedRun,
    validationSplitPercent,
    trainingMetrics,
    
    // Getters
    selectedDataset,
    activeRun,
    
    // Actions
    setSelectedDataset,
    setValidationSplitPercent,
    addDataset,
    setDatasets,
    setBaseModels,
    setPresets,
    setActiveRun,
    setCompletedRun,
    fetchCompletedRun,
    addTrainingRun,
    setTrainingRuns,
    addTrainingMetric,
    setTrainingMetrics,
    clearTrainingMetrics,
    clearTrainingState
  }
})
