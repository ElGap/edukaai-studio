<template>
  <div class="space-y-6">
    <!-- Step Navigation -->
    <nav class="border-b border-slate-800 bg-slate-900 -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8">
      <div class="flex items-center gap-1 overflow-x-auto py-2">
        <template v-for="(step, index) in steps" :key="step.id">
          <button
            v-if="isStepAccessible(step.id)"
            @click="goToStep(step.id)"
            :class="[
              'flex-shrink-0 px-4 py-2 rounded-md text-sm font-medium transition-colors',
              currentStep === step.id
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:text-white hover:bg-slate-800'
            ]"
          >
            <span class="mr-2">{{ index + 1 }}</span>
            {{ step.title }}
          </button>
          <span
            v-else
            class="flex-shrink-0 px-4 py-2 rounded-md text-sm font-medium text-slate-600 cursor-not-allowed"
          >
            <span class="mr-2">{{ index + 1 }}</span>
            {{ step.title }}
          </span>
          <span v-if="index < steps.length - 1" class="text-slate-700 mx-1">›</span>
        </template>
      </div>
    </nav>

    <!-- Step Content -->
    <div v-if="currentStep === 1">
      <DatasetsView />
    </div>
    <div v-else-if="currentStep === 2">
      <ConfigureTrainingView />
    </div>
    <div v-else-if="currentStep === 3">
      <TrainingView />
    </div>
    <div v-else-if="currentStep === 4">
      <SummaryView />
    </div>
    <div v-else-if="currentStep === 5">
      <DualChatView />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, provide, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useTrainingStore } from '@/stores/training'
import DatasetsView from '@/views/DatasetsView.vue'
import ConfigureTrainingView from '@/views/ConfigureTrainingView.vue'
import TrainingView from '@/views/TrainingView.vue'
import SummaryView from '@/views/SummaryView.vue'
import DualChatView from '@/views/DualChatView.vue'

const store = useTrainingStore()
const router = useRouter()

const currentStep = ref(1)

const steps = [
  { id: 1, title: 'Datasets' },
  { id: 2, title: 'Configure' },
  { id: 3, title: 'Training' },
  { id: 4, title: 'Summary' },
  { id: 5, title: 'Dual Chat' },
]

const isStepAccessible = (step: number): boolean => {
  if (step === 1) return true
  if (step === 2) return !!store.selectedDatasetId
  if (step === 3) return !!store.activeRunId
  if (step === 4) return !!store.completedRun
  if (step === 5) return !!store.completedRun
  return false
}

const goToStep = (step: number, data?: Record<string, any>) => {
  if (data?.selectedDatasetId) {
    store.setSelectedDataset(data.selectedDatasetId)
  }
  if (data?.activeRunId) {
    store.setActiveRun(data.activeRunId)
  }
  if (data?.completedRun) {
    store.setCompletedRun(data.completedRun)
  }
  currentStep.value = step
}

const goBack = () => {
  if (currentStep.value > 1) {
    currentStep.value -= 1
  }
}

provide('wizard', {
  currentStep,
  goToStep,
  goBack,
  isStepAccessible,
})

onMounted(() => {
  if (store.completedRun) {
    currentStep.value = 4
  } else if (store.activeRunId) {
    currentStep.value = 3
  } else if (store.selectedDatasetId) {
    currentStep.value = 2
  } else {
    currentStep.value = 1
  }
})
</script>
