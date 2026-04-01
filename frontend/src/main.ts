import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

import './assets/main.css'

console.log('[Main] Starting application...')

try {
  const app = createApp(App)
  console.log('[Main] App created')

  app.use(createPinia())
  console.log('[Main] Pinia initialized')

  app.use(router)
  console.log('[Main] Router initialized')

  app.mount('#app')
  console.log('[Main] App mounted successfully')
} catch (error) {
  console.error('[Main] Error starting application:', error)
}
