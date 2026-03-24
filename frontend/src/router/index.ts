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
      beforeEnter: (_to, _from, next) => {
        const store = useTrainingStore()
        if (!store.selectedDatasetId) {
          next({ name: 'datasets' })
        } else {
          next()
        }
      }
    },
    {
      path: '/training',
      name: 'training',
      component: () => import('@/views/TrainingView.vue'),
      meta: { step: 3, title: 'Training' },
      beforeEnter: (_to, _from, next) => {
        const store = useTrainingStore()
        if (!store.activeRunId) {
          next({ name: 'configure' })
        } else {
          next()
        }
      }
    },
    {
      path: '/summary',
      name: 'summary',
      component: () => import('@/views/SummaryView.vue'),
      meta: { step: 4, title: 'Training Summary' },
      beforeEnter: (_to, _from, next) => {
        const store = useTrainingStore()
        if (!store.completedRun) {
          next({ name: 'training' })
        } else {
          next()
        }
      }
    },
    {
      path: '/chat',
      name: 'chat',
      component: () => import('@/views/DualChatView.vue'),
      meta: { step: 5, title: 'Dual Chat' },
      beforeEnter: (_to, _from, next) => {
        const store = useTrainingStore()
        if (!store.completedRun) {
          next({ name: 'summary' })
        } else {
          next()
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
  const store = useTrainingStore()
  const toStep = to.meta.step as number | null
  const fromStep = from.meta.step as number | null
  
  console.log(`Router guard: ${String(from.name)} -> ${String(to.name)}, steps: ${fromStep} -> ${toStep}`)
  console.log(`Store state: selectedDatasetId=${store.selectedDatasetId}, activeRunId=${store.activeRunId}, completedRun=${store.completedRun?.id}`)
  
  // Allow navigation to Models page at any time
  if (to.name === 'models') {
    next()
    return
  }
  
  // STATUS CHECK: If trying to go to training page (step 3)
  if (toStep === 3) {
    // Check if we have a completed run - don't allow restarting from training page
    if (store.completedRun) {
      const completedStatus = store.completedRun.status
      if (completedStatus === 'completed') {
        console.log('BLOCKED: Training already completed. Must start new training from datasets page.')
        next({ name: 'datasets' })
        return
      }
    }
    
    // Check active run status
    const activeRun = store.activeRun
    if (activeRun) {
      // If training is running or paused, allow viewing
      if (activeRun.status === 'running' || activeRun.status === 'paused') {
        console.log('ALLOWED: Training is active, showing training page')
        next()
        return
      }
      
      // If training failed, allow retry but show warning
      if (activeRun.status === 'failed') {
        console.log('ALLOWED: Previous training failed, allowing retry')
        next()
        return
      }
      
      // If training was stopped by user, allow restart
      if (activeRun.status === 'stopped') {
        console.log('ALLOWED: Training was stopped, allowing restart')
        next()
        return
      }
      
      // If training is completed, redirect to summary
      if (activeRun.status === 'completed') {
        console.log('REDIRECT: Training completed, going to summary')
        next({ name: 'summary' })
        return
      }
    }
    
    // No active run and no completed run - must configure first
    if (!store.activeRunId && fromStep !== 2) {
      console.log('BLOCKED: No active training run, go to configure first')
      next({ name: 'configure' })
      return
    }
  }
  
  // STATUS CHECK: If trying to go to summary page (step 4)
  if (toStep === 4) {
    // Must have completed run
    if (!store.completedRun) {
      // Check if we have an active run that completed
      const activeRun = store.activeRun
      if (!activeRun || activeRun.status !== 'completed') {
        console.log('BLOCKED: No completed training run available')
        next({ name: 'training' })
        return
      }
    }
  }
  
  // STATUS CHECK: If trying to go to chat page (step 5)
  if (toStep === 5) {
    // Must have completed run with fine-tuned model
    if (!store.completedRun) {
      console.log('BLOCKED: No completed run for chat')
      next({ name: 'summary' })
      return
    }
    
    // Verify the completed run actually succeeded
    if (store.completedRun.status !== 'completed') {
      console.log('BLOCKED: Training did not complete successfully')
      next({ name: 'summary' })
      return
    }
  }
  
  // Allow backward navigation
  if (toStep === null || fromStep === null || (toStep as number) < fromStep) {
    console.log('Allowing backward navigation')
    next()
    return
  }
  
  // Check if user can proceed to this step
  if (toStep === 2 && !store.selectedDatasetId) {
    console.log('Blocking: No dataset selected')
    next({ name: 'datasets' })
    return
  }
  
  if (toStep === 3 && !store.activeRunId) {
    console.log('Blocking: No active run')
    // Allow if we just completed training and are viewing summary/chat
    // But don't allow going back to training without starting new
    if (fromStep === 4 && store.completedRun) {
      console.log('Blocking: Training already completed, start new from datasets')
      next({ name: 'datasets' })
      return
    }
    // Only allow if coming from configure (step 2) where we just started training
    if (fromStep !== 2) {
      next({ name: 'configure' })
      return
    }
    console.log('Allowing navigation to training from configure even without activeRunId (training just started)')
  }
  
  if (toStep === 4 && !store.completedRun) {
    console.log('Blocking: No completed run')
    next({ name: 'training' })
    return
  }
  
  if (toStep === 5 && !store.completedRun) {
    console.log('Blocking: No completed run for chat')
    next({ name: 'summary' })
    return
  }
  
  console.log('Navigation allowed')
  next()
})

export default router
