<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="border-b border-slate-800 pb-6">
      <h1 class="text-2xl font-bold text-white">Step 2: Configure Training</h1>
      <p class="mt-2 text-slate-400">
        Select your base model, training preset, and adjust parameters
      </p>
    </div>

    <div v-if="selectedDataset" class="bg-blue-900/20 border border-blue-800 rounded-lg p-4 mb-6">
      <p class="text-sm text-blue-300">
        <span class="font-medium">Training Dataset:</span> {{ selectedDataset.name }}
        ({{ selectedDataset.num_samples }} samples)
      </p>
    </div>

    <div v-else class="bg-red-900/20 border border-red-800 rounded-lg p-4 mb-6">
      <p class="text-sm text-red-300">
        <span class="font-medium">No dataset selected.</span> 
        Please <router-link to="/" class="underline hover:text-red-200">upload a dataset</router-link> first.
      </p>
    </div>

    <div class="grid gap-6 lg:grid-cols-3">
      <!-- Left Panel - Configuration -->
      <div class="lg:col-span-2 space-y-6">
        <!-- Base Model Selection -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 class="text-lg font-semibold text-white mb-4">Base Model</h2>
          
          <div v-if="loadingModels" class="text-center py-4">
            <div class="animate-spin h-6 w-6 border-2 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
          </div>
          
          <div v-else-if="baseModels.length === 0" class="text-slate-400 text-center py-4">
            No models available
          </div>
          
          <div v-else class="border border-slate-700 rounded-lg overflow-hidden">
            <div class="overflow-x-auto max-h-[320px]">
              <table class="w-full text-left border-collapse table-fixed">
                <thead class="bg-slate-800 sticky top-0 z-10">
                  <tr>
                    <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider w-12"></th>
                    <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider w-1/4">Model Name</th>
                    <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider w-24">Arch</th>
                    <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider w-28">Params</th>
                    <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider w-24">Context</th>
                    <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider w-28">Quant</th>
                    <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider w-24">Memory</th>
                    <th class="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right w-20">Actions</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-slate-800">
                  <!-- Curated Models -->
                  <tr
                    v-for="model in curatedModels"
                    :key="model.id"
                    @click="selectBaseModel(model)"
                    class="cursor-pointer transition-colors hover:bg-slate-800/50"
                    :class="{ 'bg-blue-900/20': selectedBaseModel?.id === model.id }"
                  >
                    <td class="px-4 py-3 w-12">
                      <input
                        type="radio"
                        :value="model.id"
                        :checked="selectedBaseModel?.id === model.id"
                        class="rounded-full border-slate-600 bg-slate-700 text-blue-600 focus:ring-blue-500"
                        @click.stop
                      />
                    </td>
                    <td class="px-4 py-3 w-1/4">
                      <div>
                        <h3 class="font-medium text-white text-sm truncate">{{ model.name }}</h3>
                        <p class="text-xs text-slate-500 mt-0.5 truncate">{{ model.huggingface_id }}</p>
                      </div>
                    </td>
                    <td class="px-4 py-3 w-24">
                      <span class="px-2 py-1 rounded text-xs font-medium bg-slate-700 text-slate-300">
                        {{ formatArchitecture(model.architecture) }}
                      </span>
                    </td>
                    <td class="px-4 py-3 w-28">
                      <span class="text-sm text-slate-300">{{ formatParameters(model.parameter_count) }}</span>
                    </td>
                    <td class="px-4 py-3 w-24">
                      <span class="text-sm text-slate-300">{{ model.context_length.toLocaleString() }}</span>
                    </td>
                    <td class="px-4 py-3 w-28">
                      <span class="text-sm text-blue-400">{{ extractQuantization(model.name) }}</span>
                    </td>
                    <td class="px-4 py-3 w-24">
                      <span class="text-sm text-orange-400">~{{ estimateMemory(model.parameter_count, model.name) }}GB</span>
                    </td>
                    <td class="px-4 py-3 w-20">
                      <!-- No actions for curated models -->
                    </td>
                  </tr>
                  
                  <!-- Custom Models -->
                  <tr
                    v-for="model in customModels"
                    :key="model.id"
                    @click="selectBaseModel(model)"
                    class="cursor-pointer transition-colors hover:bg-slate-800/50"
                    :class="{ 'bg-blue-900/20': selectedBaseModel?.id === model.id }"
                  >
                    <td class="px-4 py-3 w-12">
                      <input
                        type="radio"
                        :value="model.id"
                        :checked="selectedBaseModel?.id === model.id"
                        class="rounded-full border-slate-600 bg-slate-700 text-blue-600 focus:ring-blue-500"
                        @click.stop
                      />
                    </td>
                    <td class="px-4 py-3 w-1/4">
                      <div>
                        <h3 class="font-medium text-white text-sm truncate">{{ model.name }}</h3>
                        <p class="text-xs text-slate-500 mt-0.5 truncate">{{ model.huggingface_id }}</p>
                        <span class="inline-block mt-1 px-1.5 py-0.5 bg-purple-600/20 text-purple-400 text-[10px] rounded">Custom</span>
                      </div>
                    </td>
                    <td class="px-4 py-3 w-24">
                      <span class="px-2 py-1 rounded text-xs font-medium bg-slate-700 text-slate-300">
                        {{ formatArchitecture(model.architecture) }}
                      </span>
                    </td>
                    <td class="px-4 py-3 w-28">
                      <span class="text-sm text-slate-300">{{ formatParameters(model.parameter_count) }}</span>
                    </td>
                    <td class="px-4 py-3 w-24">
                      <span class="text-sm text-slate-300">{{ model.context_length.toLocaleString() }}</span>
                    </td>
                    <td class="px-4 py-3 w-28">
                      <span class="text-sm text-blue-400">{{ extractQuantization(model.name) }}</span>
                    </td>
                    <td class="px-4 py-3 w-24">
                      <span class="text-sm text-orange-400">~{{ estimateMemory(model.parameter_count, model.name) }}GB</span>
                    </td>
                    <td class="px-4 py-3 text-right w-20">
                      <button
                        @click.stop="confirmDeleteModel(model)"
                        class="px-2 py-1 text-xs bg-red-600/20 hover:bg-red-600/40 text-red-400 rounded transition-colors"
                        title="Delete custom model"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                  
                  <!-- Add Custom Model Row -->
                  <tr
                    @click="showCustomModelModal = true"
                    class="cursor-pointer transition-colors hover:bg-slate-800/50 border-t-2 border-dashed border-slate-600"
                  >
                    <td class="px-4 py-4 text-center" colspan="8">
                      <div class="flex items-center justify-center gap-2 text-slate-400 hover:text-slate-300">
                        <span class="text-lg">➕</span>
                        <span class="font-medium">Use Custom Model</span>
                        <span class="text-xs text-slate-500">- Any MLX-compatible HuggingFace model</span>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        
        <!-- Custom Model Input Modal -->
          <div v-if="showCustomModelModal" class="mt-4 p-4 bg-slate-800 rounded-lg border border-blue-800">
            <h3 class="text-sm font-medium text-blue-300 mb-3">Enter Custom Model</h3>
            
            <div class="space-y-3">
              <div>
                <label class="block text-xs text-slate-400 mb-1">
                  HuggingFace Model ID
                </label>
                <input
                  v-model="customModelInput"
                  type="text"
                  placeholder="e.g., mlx-community/Llama-3.2-1B-Instruct-4bit"
                  class="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-sm focus:border-blue-500 focus:outline-none"
                  @keyup.enter="validateCustomModel"
                />
                <p class="text-xs text-slate-500 mt-1">
                  Format: organization/model-name or model-name
                </p>
              </div>
              
              <!-- Validation Status -->
              <div v-if="customModelValidation" class="text-sm">
                <div v-if="customModelValidation.loading" class="flex items-center gap-2 text-blue-400">
                  <div class="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                  <span>Validating model...</span>
                </div>
                <div v-else-if="customModelValidation.error" class="text-red-400">
                  ❌ {{ customModelValidation.error }}
                </div>
                <div v-else-if="customModelValidation.success" class="text-green-400">
                  ✅ {{ customModelValidation.message }}
                  <div v-if="customModelValidation.modelInfo" class="mt-2 p-2 bg-slate-900 rounded text-xs text-slate-300">
                    <p><strong>Architecture:</strong> {{ customModelValidation.modelInfo.architecture }}</p>
                    <p><strong>Parameters:</strong> {{ (customModelValidation.modelInfo.parameter_count / 1e9).toFixed(1) }}B</p>
                    <p><strong>MLX Formatted:</strong> {{ customModelValidation.modelInfo.is_mlx_formatted ? 'Yes' : 'No (may need conversion)' }}</p>
                  </div>
                </div>
              </div>
              
              <!-- Actions -->
              <div class="flex gap-2">
                <button
                  @click="validateCustomModel"
                  :disabled="!customModelInput.trim() || customModelValidation?.loading"
                  class="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  {{ customModelValidation?.loading ? 'Validating...' : 'Validate Model' }}
                </button>
                <button
                  v-if="customModelValidation?.success"
                  @click="addCustomModel"
                  class="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Use This Model
                </button>
                <button
                  @click="showCustomModelModal = false; resetCustomModel()"
                  class="px-3 py-2 border border-slate-600 text-slate-400 hover:text-white rounded-lg text-sm transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Training Presets -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 class="text-lg font-semibold text-white mb-4">Training Preset</h2>
          
          <div v-if="loadingPresets" class="text-center py-4">
            <div class="animate-spin h-6 w-6 border-2 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
          </div>
          
          <div v-else class="flex flex-col sm:flex-row gap-4">
            <!-- Preset Selection List -->
            <div class="sm:w-1/3">
              <div class="space-y-1 max-h-[200px] overflow-y-auto pr-2">
                <div
                  v-for="preset in presets"
                  :key="preset.id"
                  @click="selectPreset(preset)"
                  :class="[
                    'p-3 rounded-lg cursor-pointer transition-all border',
                    selectedPreset?.id === preset.id
                      ? 'border-blue-500 bg-blue-900/20'
                      : 'border-slate-700 bg-slate-800 hover:border-slate-600'
                  ]"
                >
                  <div class="flex items-center justify-between">
                    <span class="font-medium text-white text-sm">{{ preset.name }}</span>
                    <span
                      v-if="preset.is_default"
                      class="px-1.5 py-0.5 rounded text-[10px] bg-blue-600 text-white"
                    >
                      Default
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- Preset Details Panel -->
            <div v-if="selectedPreset" class="sm:w-2/3 bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <div class="flex items-center justify-between mb-3">
                <h3 class="font-semibold text-white">{{ selectedPreset.name }}</h3>
                <span class="px-2 py-0.5 rounded text-xs bg-slate-700 text-slate-300">
                  {{ selectedPreset.steps }} steps
                </span>
              </div>
              
              <p v-if="selectedPreset.description" class="text-sm text-slate-400 mb-4">
                {{ selectedPreset.description }}
              </p>
              
              <div class="grid grid-cols-2 gap-3 text-sm">
                <div class="bg-slate-800 rounded p-2">
                  <span class="text-slate-500 text-xs block">Learning Rate</span>
                  <span class="text-slate-300">{{ formatScientific(selectedPreset.learning_rate) }}</span>
                </div>
                <div class="bg-slate-800 rounded p-2">
                  <span class="text-slate-500 text-xs block">LoRA Rank</span>
                  <span class="text-slate-300">r={{ selectedPreset.lora_rank }}, α={{ selectedPreset.lora_alpha }}</span>
                </div>
                <div class="bg-slate-800 rounded p-2">
                  <span class="text-slate-500 text-xs block">Batch Size</span>
                  <span class="text-slate-300">{{ selectedPreset.batch_size }}</span>
                </div>
                <div class="bg-slate-800 rounded p-2">
                  <span class="text-slate-500 text-xs block">LoRA Dropout</span>
                  <span class="text-slate-300">{{ selectedPreset.lora_dropout }}</span>
                </div>
              </div>
            </div>
            
            <div v-else class="sm:w-2/3 flex items-center justify-center bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <p class="text-slate-500 text-sm">Select a preset to view details</p>
            </div>
          </div>
        </div>

        <!-- Training Parameters -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-lg font-semibold text-white">Training Parameters</h2>
            <button
              @click="resetToPreset"
              class="text-sm text-blue-400 hover:text-blue-300"
            >
              Reset to Preset
            </button>
          </div>
          
          <div class="grid gap-6 sm:grid-cols-2">
            <!-- Steps -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                Training Steps
              </label>
              <input
                v-model.number="config.steps"
                type="number"
                min="10"
                max="10000"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              />
              <p class="text-xs text-slate-500 mt-1">
                Recommended: 100-1000 steps
              </p>
            </div>

            <!-- Learning Rate -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                Learning Rate
              </label>
              <input
                v-model="config.learning_rate"
                type="text"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              />
              <p class="text-xs text-slate-500 mt-1">
                Typical: 1e-4 (Quick) to 1e-5 (Maximum)
              </p>
            </div>

            <!-- LoRA Rank -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                LoRA Rank (r)
              </label>
              <select
                v-model.number="config.lora_rank"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option :value="4">4 (Minimal)</option>
                <option :value="8">8 (Fast)</option>
                <option :value="16">16 (Balanced)</option>
                <option :value="32">32 (Quality)</option>
                <option :value="64">64 (Maximum)</option>
                <option :value="128">128 (Extreme)</option>
              </select>
            </div>

            <!-- LoRA Alpha -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                LoRA Alpha (α)
              </label>
              <input
                v-model.number="config.lora_alpha"
                type="number"
                min="4"
                max="256"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              />
              <p class="text-xs text-slate-500 mt-1">
                Typically 2× rank
              </p>
            </div>

            <!-- Batch Size -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                Batch Size
              </label>
              <select
                v-model.number="config.batch_size"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option :value="1">1 (Low Memory)</option>
                <option :value="2">2</option>
                <option :value="4">4 (Default)</option>
                <option :value="8">8</option>
                <option :value="16">16 (High Memory)</option>
              </select>
            </div>

            <!-- Max Sequence Length -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                Max Sequence Length
              </label>
              <select
                v-model.number="config.max_seq_length"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option :value="512">512 tokens</option>
                <option :value="1024">1024 tokens</option>
                <option :value="2048">2048 tokens (Default)</option>
                <option :value="4096">4096 tokens</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Advanced Parameters -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <button
            @click="showAdvanced = !showAdvanced"
            class="flex items-center justify-between w-full text-left"
          >
            <h2 class="text-lg font-semibold text-white">Advanced Parameters</h2>
            <svg
              :class="['w-5 h-5 text-slate-400 transition-transform', showAdvanced ? 'rotate-180' : '']"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          <div v-if="showAdvanced" class="mt-4 grid gap-6 sm:grid-cols-2">
            <!-- Gradient Checkpointing -->
            <div class="flex items-start gap-3">
              <input
                v-model="config.gradient_checkpointing"
                type="checkbox"
                id="grad-checkpoint"
                class="mt-1 w-4 h-4 rounded border-slate-600 bg-slate-800 text-blue-600"
              />
              <div>
                <label for="grad-checkpoint" class="font-medium text-white cursor-pointer">
                  Gradient Checkpointing
                </label>
                <p class="text-xs text-slate-400 mt-1">
                  Reduces memory by 30-50%, slower training
                </p>
              </div>
            </div>

            <!-- Number of LoRA Layers -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                LoRA Layers ({{ config.num_lora_layers }})
              </label>
              <input
                v-model.number="config.num_lora_layers"
                type="range"
                min="4"
                max="32"
                step="4"
                class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
              <p class="text-xs text-slate-500 mt-1">
                Fewer layers = faster, more memory
              </p>
            </div>

            <!-- Prompt Masking -->
            <div class="flex items-start gap-3">
              <input
                v-model="config.prompt_masking"
                type="checkbox"
                id="prompt-mask"
                class="mt-1 w-4 h-4 rounded border-slate-600 bg-slate-800 text-blue-600"
              />
              <div>
                <label for="prompt-mask" class="font-medium text-white cursor-pointer">
                  Prompt Masking
                </label>
                <p class="text-xs text-slate-400 mt-1">
                  Only train on assistant responses (recommended for chat)
                </p>
              </div>
            </div>

            <!-- Warmup Steps -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                Warmup Steps
              </label>
              <input
                v-model.number="config.warmup_steps"
                type="number"
                min="0"
                max="500"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              />
            </div>

            <!-- Gradient Accumulation -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                Gradient Accumulation Steps
              </label>
              <input
                v-model.number="config.gradient_accumulation_steps"
                type="number"
                min="1"
                max="8"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              />
              <p class="text-xs text-slate-500 mt-1">
                Effective batch = batch_size × accumulation
              </p>
            </div>

            <!-- Early Stopping -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                Early Stopping Patience
              </label>
              <input
                v-model.number="config.early_stopping_patience"
                type="number"
                min="0"
                max="50"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              />
              <p class="text-xs text-slate-500 mt-1">
                0 = disabled
              </p>
            </div>

            <!-- Validation Split -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                Validation Split ({{ config.validation_split_percent }}%)
              </label>
              <div class="flex gap-2">
                <button
                  v-for="percent in [5, 10, 15]"
                  :key="percent"
                  @click="config.validation_split_percent = percent"
                  :class="[
                    'px-3 py-2 rounded-lg text-sm font-medium transition-colors flex-1',
                    config.validation_split_percent === percent
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                  ]"
                >
                  {{ percent }}%
                </button>
              </div>
              <p class="text-xs text-slate-500 mt-1">
                Percentage of data used for validation
              </p>
              <p class="text-xs text-slate-400 mt-1">
                <span v-if="config.validation_split_percent === 5" class="text-green-400">
                  ✓ More training data (95%), good for large datasets
                </span>
                <span v-else-if="config.validation_split_percent === 10" class="text-blue-400">
                  ✓ Balanced - standard choice (90% train)
                </span>
                <span v-else-if="config.validation_split_percent === 15" class="text-orange-400">
                  ✓ More validation data (85%), better for small datasets
                </span>
              </p>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Panel - Resource Limits & Summary -->
      <div class="space-y-6">
        <!-- Resource Limits -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 class="text-lg font-semibold text-white mb-4">Resource Limits</h2>
          
          <div class="space-y-4">
            <!-- CPU Cores -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                CPU Cores ({{ resourceLimits.cpu_cores || 'All' }})
              </label>
              <input
                v-model.number="resourceLimits.cpu_cores"
                type="range"
                min="1"
                max="8"
                class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
              <p class="text-xs text-slate-500 mt-1">
                Leave at max for best performance
              </p>
            </div>

            <!-- GPU Memory -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                GPU Memory Limit
              </label>
              <select
                v-model.number="resourceLimits.gpu_memory_gb"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option :value="null">Unlimited</option>
                <option :value="4">4 GB</option>
                <option :value="8">8 GB</option>
                <option :value="12">12 GB</option>
                <option :value="16">16 GB</option>
              </select>
            </div>

            <!-- System RAM -->
            <div>
              <label class="block text-sm font-medium text-slate-300 mb-2">
                System RAM Limit
              </label>
              <select
                v-model.number="resourceLimits.ram_gb"
                class="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option :value="null">Unlimited</option>
                <option :value="8">8 GB</option>
                <option :value="16">16 GB</option>
                <option :value="24">24 GB</option>
                <option :value="32">32 GB</option>
              </select>
            </div>
          </div>

          <div class="mt-4 p-3 bg-yellow-900/20 border border-yellow-800 rounded-lg">
            <p class="text-xs text-yellow-300">
              <span class="font-medium">💡 Tip:</span> Lower limits prevent system slowdown but may cause training to fail on large models.
            </p>
          </div>
        </div>

        <!-- Configuration Summary -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h2 class="text-lg font-semibold text-white mb-4">Summary</h2>
          
          <div class="space-y-3 text-sm">
            <div class="flex justify-between">
              <span class="text-slate-400">Run Name</span>
              <span class="text-white">{{ runName }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Model</span>
              <span class="text-white">{{ selectedBaseModel?.name || 'Not selected' }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Preset</span>
              <span class="text-white">{{ selectedPreset?.name || 'Not selected' }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Steps</span>
              <span class="text-white">{{ config.steps }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">LoRA</span>
              <span class="text-white">r={{ config.lora_rank }}, α={{ config.lora_alpha }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Batch Size</span>
              <span class="text-white">{{ config.batch_size }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Est. Time</span>
              <span class="text-white">~{{ estimatedTime }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Actions -->
    <div class="flex items-center justify-between pt-6 border-t border-slate-800">
      <router-link
        to="/"
        class="px-6 py-3 text-slate-400 hover:text-white transition-colors"
      >
        ← Back to Datasets
      </router-link>
      
      <button
        @click="startTraining"
        :disabled="!canStartTraining || isStartingTraining"
        :class="[
          'px-8 py-3 rounded-lg font-medium transition-all flex items-center gap-2',
          canStartTraining && !isStartingTraining
            ? 'bg-blue-600 hover:bg-blue-700 text-white'
            : 'bg-slate-700 text-slate-500 cursor-not-allowed'
        ]"
      >
        <svg v-if="isStartingTraining" class="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
         {{ isStartingTraining ? 'Starting Training...' : (canStartTraining ? 'Start Training →' : 'Select Model & Dataset First') }}
       </button>
     </div>
  </div>

  <!-- Delete Custom Model Confirmation Modal -->
  <div
    v-if="showDeleteModelModal"
    class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
    @click="showDeleteModelModal = false"
  >
    <div
      class="bg-slate-900 rounded-xl border border-slate-800 p-6 w-full max-w-md mx-4"
      @click.stop
    >
      <h3 class="text-lg font-semibold text-white mb-4">Delete Custom Model</h3>
      
      <p class="text-slate-400 mb-4">
        Are you sure you want to delete <strong class="text-white">{{ modelToDelete?.name }}</strong>?
      </p>
      
      <div class="bg-slate-800 rounded-lg p-3 mb-4">
        <p class="text-xs text-slate-400">Model ID:</p>
        <p class="text-sm text-slate-300">{{ modelToDelete?.huggingface_id }}</p>
      </div>
      
      <p class="text-xs text-slate-500 mb-6">
        This will remove the model from the list. If the model has been used in training runs, it will be hidden but existing runs will continue to work.
      </p>
      
      <div class="flex justify-end gap-3">
        <button
          @click="showDeleteModelModal = false; modelToDelete = null"
          class="px-4 py-2 text-slate-400 hover:text-white transition-colors"
          :disabled="isDeletingModel"
        >
          Cancel
        </button>
        <button
          @click="deleteModel"
          :disabled="isDeletingModel"
          class="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-800/50 text-white rounded-lg font-medium transition-colors"
        >
          <span v-if="isDeletingModel">Deleting...</span>
          <span v-else>Delete Model</span>
        </button>
      </div>
    </div>
    
    <!-- Delete Custom Model Confirmation Modal -->
    <div
      v-if="showDeleteModelModal"
      class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      @click="showDeleteModelModal = false"
    >
      <div
        class="bg-slate-900 rounded-xl border border-slate-800 p-6 w-full max-w-md mx-4"
        @click.stop
      >
        <h3 class="text-lg font-semibold text-white mb-4">Delete Custom Model</h3>
        
        <p class="text-slate-400 mb-4">
          Are you sure you want to delete <strong class="text-white">{{ modelToDelete?.name }}</strong>?
        </p>
        
        <div class="bg-slate-800 rounded-lg p-3 mb-4">
          <p class="text-xs text-slate-400">Model ID:</p>
          <p class="text-sm text-slate-300">{{ modelToDelete?.huggingface_id }}</p>
        </div>
        
        <p class="text-xs text-slate-500 mb-6">
          This will remove the model from the list. If the model has been used in training runs, it will be hidden but existing runs will continue to work.
        </p>
        
        <div class="flex justify-end gap-3">
          <button
            @click="showDeleteModelModal = false; modelToDelete = null"
            class="px-4 py-2 text-slate-400 hover:text-white transition-colors"
            :disabled="isDeletingModel"
          >
            Cancel
          </button>
          <button
            @click="deleteModel"
            :disabled="isDeletingModel"
            class="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-800/50 text-white rounded-lg font-medium transition-colors"
          >
            <span v-if="isDeletingModel">Deleting...</span>
            <span v-else>Delete Model</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useTrainingStore } from '@/stores/training'
import type { BaseModel, TrainingPreset } from '@/stores/training'
import axios from 'axios'
import { useToast } from '@/composables/useToast'

const router = useRouter()
const store = useTrainingStore()
const { error: showError, success: showSuccess } = useToast()

// API client
const api = axios.create({
  baseURL: '/api'
})

// State
const baseModels = ref<BaseModel[]>([])
const presets = ref<TrainingPreset[]>([])
const selectedBaseModel = ref<BaseModel | null>(null)
const selectedPreset = ref<TrainingPreset | null>(null)
const loadingModels = ref(false)
const loadingPresets = ref(false)
const isStartingTraining = ref(false)
const showAdvanced = ref(false)

// Custom model state
const showCustomModelModal = ref(false)
const customModelInput = ref('')
const customModelValidation = ref<{
  loading: boolean
  success: boolean
  error: string
  message: string
  modelInfo?: {
    architecture: string
    parameter_count: number
    context_length: number
    is_mlx_formatted: boolean
    tags: string[]
  }
} | null>(null)

// Delete custom model state
const showDeleteModelModal = ref(false)
const modelToDelete = ref<BaseModel | null>(null)
const isDeletingModel = ref(false)

// Configuration
const config = ref({
  steps: 100,
  learning_rate: 0.0001,
  lora_rank: 8,
  lora_alpha: 16,
  lora_dropout: 0.05,
  batch_size: 4,
  max_seq_length: 2048,
  warmup_steps: 10,
  gradient_accumulation_steps: 1,
  early_stopping_patience: 0,
  gradient_checkpointing: false,
  num_lora_layers: 8,
  prompt_masking: true,
  validation_split_percent: store.validationSplitPercent || 10  // Read from store or default 10%
})

// Resource limits
const resourceLimits = ref({
  cpu_cores: null as number | null,
  gpu_memory_gb: null as number | null,
  ram_gb: null as number | null
})

// Getters
const selectedDataset = computed(() => store.selectedDataset)

const curatedModels = computed(() => {
  return baseModels.value.filter(m => m.mlx_config?.is_curated !== false && m.mlx_config?.is_custom !== true)
})

const customModels = computed(() => {
  return baseModels.value.filter(m => m.mlx_config?.is_custom === true)
})

const runName = computed(() => {
  if (!selectedBaseModel.value) return 'Not configured'
  const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '')
  return `${selectedBaseModel.value.name.replace(/\s+/g, '')}-${timestamp}`
})

const canStartTraining = computed(() => {
  const canStart = !!(selectedBaseModel.value && selectedPreset.value && selectedDataset.value)
  console.log('canStartTraining check:', { 
    hasModel: !!selectedBaseModel.value, 
    hasPreset: !!selectedPreset.value, 
    hasDataset: !!selectedDataset.value,
    canStart 
  })
  return canStart
})

const estimatedTime = computed(() => {
  if (!selectedBaseModel.value || !config.value.steps) return 'Unknown'
  
  // Rough estimate: 250 tokens/sec on M1 Max, ~500 samples
  // This is a very rough estimate - real time depends on model size
  const estimatedSeconds = config.value.steps * 2  // ~2 seconds per step rough estimate
  const minutes = Math.ceil(estimatedSeconds / 60)
  
  if (minutes < 60) {
    return `${minutes} min`
  } else {
    const hours = Math.floor(minutes / 60)
    const remainingMinutes = minutes % 60
    return `${hours}h ${remainingMinutes}m`
  }
})

// Methods
const formatParameters = (count: number) => {
  if (count >= 1_000_000_000) {
    return `${(count / 1_000_000_000).toFixed(1)}B parameters`
  } else if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M parameters`
  }
  return `${count} parameters`
}

const formatScientific = (num: number) => {
  if (num >= 0.001) {
    return num.toFixed(4)
  }
  return num.toExponential(1)
}

const formatArchitecture = (arch: string): string => {
  const map: Record<string, string> = {
    'llama': 'Llama 3',
    'qwen2': 'Qwen 2.5',
    'phi3': 'Phi-3',
    'gemma': 'Gemma 2',
    'mistral': 'Mistral'
  }
  return map[arch] || arch.charAt(0).toUpperCase() + arch.slice(1)
}

const extractQuantization = (modelName: string): string => {
  const match = modelName.match(/(\d+)-bit/)
  return match ? `${match[1]}-bit` : 'Unknown'
}

const estimateMemory = (params: number, modelName: string): number => {
  // Extract quantization bits from name (default to 16)
  const quantMatch = modelName.match(/(\d+)-bit/)
  const bits = quantMatch ? parseInt(quantMatch[1]) : 16
  
  // Calculate base model memory: params * bits / 8 / 1024^3
  const baseMemoryGB = (params * bits) / (8 * 1024 * 1024 * 1024)
  
  // Add overhead for LoRA (~50MB = 0.05GB) and activations (~0.5GB)
  const overheadGB = 0.55
  
  return Math.ceil(baseMemoryGB + overheadGB)
}

const selectBaseModel = (model: BaseModel) => {
  console.log('[MODEL SELECT] User selected model:', model.id, model.name, model.huggingface_id)
  selectedBaseModel.value = model
  showCustomModelModal.value = false  // Close modal if open
}

const confirmDeleteModel = (model: BaseModel) => {
  modelToDelete.value = model
  showDeleteModelModal.value = true
}

const deleteModel = async () => {
  if (!modelToDelete.value) return
  
  isDeletingModel.value = true
  
  try {
    await api.delete(`/base-models/${modelToDelete.value.id}`)
    
    // If the deleted model was selected, deselect it
    if (selectedBaseModel.value?.id === modelToDelete.value.id) {
      selectedBaseModel.value = null
    }
    
    // Refresh the models list
    const response = await api.get('/base-models')
    baseModels.value = response.data
    store.setBaseModels(response.data)
    
    showSuccess(`Deleted custom model: ${modelToDelete.value.name}`)
    showDeleteModelModal.value = false
    modelToDelete.value = null
  } catch (err: any) {
    console.error('Failed to delete model:', err)
    showError(err.response?.data?.detail || 'Failed to delete custom model')
  } finally {
    isDeletingModel.value = false
  }
}

const resetCustomModel = () => {
  customModelInput.value = ''
  customModelValidation.value = null
}

const validateCustomModel = async () => {
  if (!customModelInput.value.trim()) return
  
  customModelValidation.value = {
    loading: true,
    success: false,
    error: '',
    message: 'Validating model...'
  }
  
  try {
    const response = await api.post('/base-models/validate', {
      huggingface_id: customModelInput.value.trim()
    })
    
    const data = response.data
    
    customModelValidation.value = {
      loading: false,
      success: data.is_valid,
      error: data.is_valid ? '' : data.message,
      message: data.message,
      modelInfo: data.model_info
    }
  } catch (error: any) {
    customModelValidation.value = {
      loading: false,
      success: false,
      error: error.response?.data?.detail || 'Failed to validate model',
      message: ''
    }
  }
}

const addCustomModel = async () => {
  if (!customModelValidation.value?.success) return
  
  try {
    const response = await api.post('/base-models/custom', {
      huggingface_id: customModelInput.value.trim()
    })
    
    const newModel = response.data
    
    // Add to store and list
    baseModels.value.push(newModel)
    
    // Select the new model
    selectedBaseModel.value = newModel
    
    // Show success
    const { success: showSuccess } = useToast()
    showSuccess(`Custom model "${newModel.name}" added successfully!`)
    
    // Close modal and reset
    showCustomModelModal.value = false
    resetCustomModel()
    
  } catch (error: any) {
    const { error: showError } = useToast()
    showError(error.response?.data?.detail || 'Failed to add custom model')
  }
}

const selectPreset = (preset: TrainingPreset) => {
  selectedPreset.value = preset
  resetToPreset()
}

const resetToPreset = () => {
  if (!selectedPreset.value) return
  
  const p = selectedPreset.value
  config.value = {
    steps: p.steps,
    learning_rate: p.learning_rate,
    lora_rank: p.lora_rank,
    lora_alpha: p.lora_alpha,
    lora_dropout: p.lora_dropout ?? 0.05,
    batch_size: p.batch_size,
    max_seq_length: 2048,
    warmup_steps: p.warmup_steps ?? 10,
    gradient_accumulation_steps: p.gradient_accumulation_steps ?? 1,
    early_stopping_patience: p.early_stopping_patience ?? 0,
    gradient_checkpointing: p.gradient_checkpointing ?? false,
    num_lora_layers: p.num_lora_layers ?? 16,
    prompt_masking: p.prompt_masking ?? true,
    validation_split_percent: config.value.validation_split_percent  // Preserve user's choice
  }
}

const startTraining = async () => {
  console.log('Start Training clicked')
  console.log('canStartTraining:', canStartTraining.value)
  console.log('selectedDataset:', selectedDataset.value)
  console.log('selectedBaseModel:', selectedBaseModel.value)
  console.log('selectedPreset:', selectedPreset.value)
  
  if (!canStartTraining.value) {
    console.log('Cannot start - missing model or preset')
    showError('Please select a base model and training preset first.')
    return
  }
  
  if (!selectedDataset.value) {
    console.log('Cannot start - no dataset selected')
    showError('No dataset selected. Please go back and upload a dataset first.')
    return
  }
  
  isStartingTraining.value = true
  
  try {
    // Create training run via API
    const runData = {
      name: runName.value,
      training_dataset_id: selectedDataset.value.id,
      base_model_id: selectedBaseModel.value!.id,
      preset_id: selectedPreset.value!.id,
      validation_split_percent: config.value.validation_split_percent,
      steps: config.value.steps,
      learning_rate: config.value.learning_rate,
      lora_rank: config.value.lora_rank,
      lora_alpha: config.value.lora_alpha,
      lora_dropout: config.value.lora_dropout,
      batch_size: config.value.batch_size,
      max_seq_length: config.value.max_seq_length,
      warmup_steps: config.value.warmup_steps,
      gradient_accumulation_steps: config.value.gradient_accumulation_steps,
      early_stopping_patience: config.value.early_stopping_patience,
      gradient_checkpointing: config.value.gradient_checkpointing,
      num_lora_layers: config.value.num_lora_layers,
      prompt_masking: config.value.prompt_masking,
      cpu_cores_limit: resourceLimits.value.cpu_cores,
      gpu_memory_limit_gb: resourceLimits.value.gpu_memory_gb,
      ram_limit_gb: resourceLimits.value.ram_gb
    }
    
    console.log('[TRAINING START] Creating training run with config:', JSON.stringify(runData, null, 2))
    console.log('[TRAINING START] Base model being sent:', runData.base_model_id)
    
    // Call API to create training run
    const response = await api.post('/training/runs', runData)
    
    const result = response.data
    console.log('Training run created, full response data:', JSON.stringify(result, null, 2))
    console.log('Run ID from result:', result?.id)
    console.log('Run name from result:', result?.name)
    console.log('Run status from result:', result?.status)
    
    if (!result || !result.id) {
      throw new Error('Invalid response from server: missing run ID')
    }
    
    // Store the run ID and add complete run data to store
    store.setActiveRun(result.id)
    
    // Add the complete run data from API response to trainingRuns using store action
    store.addTrainingRun(result)
    
    console.log('Active run set in store:', store.activeRunId)
    console.log('Training run added to store, total runs:', store.trainingRuns.length)
    
    showSuccess(`Training run "${result.name}" created! Starting training...`)
    
    // Navigate immediately to training page
    console.log('About to navigate to training page...')
    
    // Use both router and window.location as fallbacks
    const navigateToTraining = async () => {
      try {
        await router.push({ name: 'training' })
        console.log('Navigation successful via router')
      } catch (err) {
        console.error('Router navigation failed:', err)
        console.log('Falling back to window.location...')
        window.location.assign('/training')
      }
    }
    
    // Small delay then navigate
    setTimeout(() => {
      navigateToTraining()
    }, 50)
  } catch (error: any) {
    console.error('Failed to create training run:', error)
    const errorMessage = error.response?.data?.detail || error.message || 'Unknown error'
    showError('Failed to start training: ' + errorMessage)
  } finally {
    isStartingTraining.value = false
  }
}

// Load data on mount
onMounted(async () => {
  loadingModels.value = true
  loadingPresets.value = true
  
  try {
    // Load base models from API
    const modelsResponse = await api.get('/base-models')
    baseModels.value = modelsResponse.data
    store.setBaseModels(modelsResponse.data)
    
    // Load training presets from API
    const presetsResponse = await api.get('/training-presets')
    presets.value = presetsResponse.data
    store.setPresets(presetsResponse.data)
    
    // Auto-select first model and preset only if none selected
    if (baseModels.value.length > 0 && !selectedBaseModel.value) {
      console.log('[AUTO SELECT] No model selected, auto-selecting first:', baseModels.value[0].id, baseModels.value[0].name)
      selectedBaseModel.value = baseModels.value[0]
    } else if (selectedBaseModel.value) {
      console.log('[AUTO SELECT] Keeping existing model selection:', selectedBaseModel.value.id, selectedBaseModel.value.name)
    }
    
    if (presets.value.length > 0 && !selectedPreset.value) {
      const defaultPreset = presets.value.find(p => p.is_default) || presets.value[0]
      console.log('[AUTO SELECT] Auto-selecting preset:', defaultPreset.id, defaultPreset.name)
      selectPreset(defaultPreset)
    }
  } catch (error: any) {
    console.error('Failed to load configuration data:', error)
    showError('Failed to load models and presets: ' + (error.response?.data?.detail || error.message))
  } finally {
    loadingModels.value = false
    loadingPresets.value = false
  }
})
</script>
