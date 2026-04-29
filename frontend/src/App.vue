<template>
  <div class="min-h-screen bg-slate-950 text-slate-100">
    <!-- Toast Notifications -->
    <Toast />

    <!-- Navigation Header -->
    <nav class="border-b border-slate-800 bg-slate-900">
      <div class="w-full px-4 sm:px-6 lg:px-8">
        <div class="flex h-16 items-center justify-between">
          <router-link to="/" class="flex items-center gap-3">
            <span class="font-semibold text-xl">EdukaAI Studio</span>
            <span class="px-2 py-0.5 text-xs font-medium bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded">BETA</span>
          </router-link>

          <div class="flex items-center gap-3">
            <router-link
              to="/models"
              class="px-4 py-2 rounded-md text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
            >
              My Models
            </router-link>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="w-full px-4 sm:px-6 lg:px-8 py-8">
      <router-view />
    </main>

    <footer class="w-full px-4 sm:px-6 lg:px-8 py-4 border-t border-slate-800 bg-slate-900 text-xs text-center text-slate-400">
      Open Source project by <a href="https://elgap.ai" class="text-blue-400 hover:text-blue-300 transition-colors" target="_blank">ElGap</a> |
      <a href="https://github.com/elgap/edukaai-studio" class="text-blue-400 hover:text-blue-300 transition-colors" target="_blank">GitHub</a>
    </footer>
  </div>
</template>

<script setup lang="ts">
import Toast from '@/components/Toast.vue'
</script>
