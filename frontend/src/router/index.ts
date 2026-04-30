import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'wizard',
      component: () => import('@/views/WizardView.vue'),
      meta: { title: 'EdukaAI Studio' }
    },
    {
      path: '/datasets',
      name: 'datasets',
      component: () => import('@/views/DatasetsView.vue'),
      meta: { title: 'Datasets' }
    },
    {
      path: '/training',
      name: 'training',
      component: () => import('@/views/TrainingView.vue'),
      meta: { title: 'Training' }
    },
    {
      path: '/configure-training',
      name: 'configure-training',
      component: () => import('@/views/ConfigureTrainingView.vue'),
      meta: { title: 'Configure Training' }
    },
    {
      path: '/models',
      name: 'models',
      component: () => import('@/views/MyModelsView.vue'),
      meta: { title: 'My Models' }
    },
    {
      path: '/summary',
      name: 'summary',
      component: () => import('@/views/SummaryView.vue'),
      meta: { title: 'Summary' }
    },
    {
      path: '/chat',
      name: 'chat',
      component: () => import('@/views/DualChatView.vue'),
      meta: { title: 'Chat' }
    }
  ]
})

router.beforeEach((to, _from, next) => {
  if (to.matched.length === 0) {
    next('/')
  } else {
    if (to.meta.title) {
      document.title = to.meta.title as string
    }
    next()
  }
})

export default router
