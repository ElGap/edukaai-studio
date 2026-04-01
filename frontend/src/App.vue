<template>
  <div class="min-h-screen bg-slate-950 text-slate-100">
    <!-- Toast Notifications -->
    <Toast />
    
    <!-- Navigation Header -->
    <nav class="border-b border-slate-800 bg-slate-900">
      <div class="w-full px-4 sm:px-6 lg:px-8">
        <div class="flex h-16 items-center justify-between">
          <!-- Logo -->
          <div class="flex items-center gap-3">
            <span class="font-semibold text-xl">EdukaAI Studio</span>
            <span class="px-2 py-0.5 text-xs font-medium bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded">BETA</span>
          </div>
          
          <!-- Navigation Steps -->
          <div class="hidden md:flex items-center gap-1">
            <router-link
              v-for="(step, index) in steps"
              :key="step.name"
              :to="getStepRoute(step)"
              :class="[
                'px-4 py-2 rounded-md text-sm font-medium transition-colors',
                isStepActive(step.name)
                  ? 'bg-blue-600 text-white'
                  : isStepEnabled(step.step)
                    ? 'text-slate-300 hover:text-white hover:bg-slate-800'
                    : 'text-slate-600 cursor-not-allowed'
              ]"
            >
              <span class="mr-2">{{ index + 1 }}</span>
              {{ step.title }}
            </router-link>
          </div>
          
          <!-- Right side actions -->
          <div class="flex items-center gap-3">
            <router-link
              to="/models"
              class="px-4 py-2 rounded-md text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
            >
              My Models
            </router-link>
            
            <!-- Theme Toggle -->
            <button
              @click="toggleTheme"
              class="p-2 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <svg v-if="isDark" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
              <svg v-else class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
    
    <!-- Mobile Navigation -->
    <div class="md:hidden border-b border-slate-800 bg-slate-900">
      <div class="flex overflow-x-auto px-4 py-2 gap-2">
        <router-link
          v-for="(step, index) in steps"
          :key="step.name"
          :to="getStepRoute(step)"
          :class="[
            'flex-shrink-0 px-3 py-1.5 rounded-md text-sm font-medium whitespace-nowrap transition-colors',
            isStepActive(step.name)
              ? 'bg-blue-600 text-white'
              : isStepEnabled(step.step)
                ? 'text-slate-300'
                : 'text-slate-600'
          ]"
        >
          {{ index + 1 }}. {{ step.title }}
        </router-link>
      </div>
    </div>
    
    <!-- Main Content -->
    <main class="w-full px-4 sm:px-6 lg:px-8 py-8">
      <router-view v-slot="{ Component, route }">
        <transition name="fade" mode="out-in">
          <div :key="route.path" class="w-full">
            <component :is="Component" />
          </div>
        </transition>
      </router-view>
    </main>

    <footer class="w-full px-4 sm:px-6 lg:px-8 py-4 border-t border-slate-800 bg-slate-900 text-xs text-center text-slate-400">
      Project by <a href="https://elgap.rs" class="text-blue-400 hover:text-blue-300 transition-colors" target="_blank">ElGap</a> | 
      <a href="https://github.com/elgap/edukaai-studio" class="text-blue-400 hover:text-blue-300 transition-colors" target="_blank">Open Source (MIT)</a> | 
      Powered by <a href="https://elgap.rs/rapid-mvp-mindset" class="text-blue-400 hover:text-blue-300 transition-colors" target="_blank">#RapidMvpMindset</a>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRoute } from 'vue-router'
import { useTrainingStore } from '@/stores/training'
import Toast from '@/components/Toast.vue'

const route = useRoute()
const store = useTrainingStore()

const isDark = ref(true)

const steps = [
  { name: 'datasets', title: 'Datasets', step: 1 },
  { name: 'configure', title: 'Configure', step: 2 },
  { name: 'training', title: 'Training', step: 3 },
  { name: 'summary', title: 'Summary', step: 4 },
  { name: 'chat', title: 'Dual Chat', step: 5 }
]

const toggleTheme = () => {
  isDark.value = !isDark.value
  document.documentElement.classList.toggle('dark')
}

const isStepActive = (name: string) => {
  return route.name === name
}

const isStepEnabled = (step: number) => {
  // Logic to determine if step is enabled based on current state
  if (step === 1) return true
  if (step === 2) return !!store.selectedDatasetId
  if (step === 3) return !!store.activeRunId
  if (step === 4) return !!store.completedRun
  if (step === 5) return !!store.completedRun
  return false
}

const getStepRoute = (step: typeof steps[0]) => {
  // Always return a valid route object
  // The router guards will handle validation and redirects
  return { name: step.name }
}
</script>

<style>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
