import { createRouter, createWebHistory } from 'vue-router'
import { useTrainingStore } from '@/stores/training'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'datasets',
      component: () => import('@/views/DatasetsView.vue'),
      meta: { step: 1, title: 'Datasets' }
    },
    {
      path: '/configure',
      name: 'configure',
      component: () => import('@/views/ConfigureTrainingView.vue'),
      meta: { step: 2, title: 'Configure Training' },
      beforeEnter: async (to, _from, next) => {
        const store = useTrainingStore()
        const datasetId = to.query.datasetId as string
        
        // Restore from URL if present
        if (datasetId) {
          store.setSelectedDataset(datasetId)
          next()
        } else if (store.selectedDatasetId) {
          // Has state but no URL param - redirect with param
          next({ name: 'configure', query: { datasetId: store.selectedDatasetId } })
        } else {
          // No dataset selected
          next({ name: 'datasets' })
        }
      }
    },
    {
      path: '/training/:runId?',
      name: 'training',
      component: () => import('@/views/TrainingView.vue'),
      meta: { step: 3, title: 'Training' },
      beforeEnter: async (to, _from, next) => {
        const store = useTrainingStore()
        const runId = to.params.runId as string
        
        // If no runId in URL, check store
        if (!runId) {
          if (store.activeRunId) {
            // Redirect to URL with runId
            next({ name: 'training', params: { runId: store.activeRunId } })
            return
          }
          // No active run - must configure first
          next({ name: 'configure' })
          return
        }
        
        // Fetch run from DB to verify it exists and get status
        try {
          const run = await store.fetchCompletedRun(runId)
          if (!run) {
            next({ name: 'datasets' })
            return
          }
          
          // Set active run for the view
          store.setActiveRun(runId)
          
          // If already completed, redirect to summary
          if (run.status === 'completed') {
            next({ name: 'summary', params: { runId } })
            return
          }
          
          next()
        } catch {
          next({ name: 'datasets' })
        }
      }
    },
    {
      path: '/summary/:runId?',
      name: 'summary',
      component: () => import('@/views/SummaryView.vue'),
      meta: { step: 4, title: 'Training Summary' },
      beforeEnter: async (to, _from, next) => {
        const store = useTrainingStore()
        const runId = to.params.runId as string
        
        // If no runId in URL, check store
        if (!runId) {
          if (store.completedRun) {
            next({ name: 'summary', params: { runId: store.completedRun.id } })
            return
          }
          next({ name: 'training' })
          return
        }
        
        try {
          const run = await store.fetchCompletedRun(runId)
          if (!run || run.status !== 'completed') {
            next({ name: 'training', params: { runId } })
            return
          }
          
          store.setCompletedRun(run)
          next()
        } catch {
          next({ name: 'datasets' })
        }
      }
    },
    {
      path: '/chat/:runId?',
      name: 'chat',
      component: () => import('@/views/DualChatView.vue'),
      meta: { step: 5, title: 'Dual Chat' },
      beforeEnter: async (to, _from, next) => {
        const store = useTrainingStore()
        const runId = to.params.runId as string
        
        // If no runId in URL, check store
        if (!runId) {
          if (store.completedRun) {
            next({ name: 'chat', params: { runId: store.completedRun.id } })
            return
          }
          next({ name: 'summary' })
          return
        }
        
        try {
          const run = await store.fetchCompletedRun(runId)
          if (!run || run.status !== 'completed') {
            next({ name: 'summary', params: { runId } })
            return
          }
          
          store.setCompletedRun(run)
          next()
        } catch {
          next({ name: 'datasets' })
        }
      }
    },
    {
      path: '/models',
      name: 'models',
      component: () => import('@/views/MyModelsView.vue'),
      meta: { step: null, title: 'My Models' }
    }
  ]
})

// Navigation guard for step progression
router.beforeEach((to, from, next) => {
  try {
    const store = useTrainingStore()
    const toStep = to.meta.step as number | null
    const fromStep = from.meta.step as number | null
    
    console.log(`[Router] Navigating: ${String(from.name)} -> ${String(to.name)}, steps: ${fromStep} -> ${toStep}`)
    
    // Allow initial load and navigation to Models page at any time
    if (to.name === 'models') {
      next()
      return
    }
    
    // Allow navigation to datasets (step 1) always
    if (toStep === 1) {
      next()
      return
    }
    
    // Allow backward navigation without restrictions
    if (toStep === null || fromStep === null || (toStep as number) <= fromStep) {
      next()
      return
    }
    
    // Forward navigation checks
    if (toStep === 2 && !store.selectedDatasetId) {
      console.log('[Router] Blocking: No dataset selected')
      next({ name: 'datasets' })
      return
    }
    
    if (toStep === 3 && !store.activeRunId) {
      console.log('[Router] Blocking: No active run')
      next({ name: 'configure' })
      return
    }
    
    if (toStep === 4 && !store.completedRun) {
      console.log('[Router] Blocking: No completed run')
      next({ name: 'training' })
      return
    }
    
    if (toStep === 5 && !store.completedRun) {
      console.log('[Router] Blocking: No completed run for chat')
      next({ name: 'summary' })
      return
    }
    
    next()
  } catch (error) {
    console.error('[Router] Error in navigation guard:', error)
    next({ name: 'datasets' })
  }
})

export default router
