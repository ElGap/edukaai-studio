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
      path: '/models',
      name: 'models',
      component: () => import('@/views/MyModelsView.vue'),
      meta: { title: 'My Models' }
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
