// Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

import axiosInstance from '@renderer/services/axiosConfig'
import { get } from 'lodash'

export interface BasicModelConfig {
  modelPlatform: string // Model platform, e.g., doubao, openai, custom
  modelId: string // Model ID
  baseUrl: string // API base URL
  apiKey: string // API key
  provider?: string // Provider name (for backend compatibility)
  outputDim?: number // Optional embedding dimension override
}

// Model configuration interface
export interface ModelConfigProps {
  vlm: BasicModelConfig
  llm: BasicModelConfig
  embedding: BasicModelConfig
  reranker?: BasicModelConfig
}

// API response data structure
export interface ModelInfoResponseData {
  config: ModelConfigProps
}

// Complete API response structure
export interface ApiResponse<T> {
  code: number
  status: number
  message: string
  data: T
}

// Get model settings information
export const getModelInfo = async (): Promise<ModelInfoResponseData | undefined> => {
  const res = await axiosInstance.get<ModelInfoResponseData>('/api/model_settings/get')
  return get(res, 'data.data')
}

// Update model settings information
// Update model settings information response data structure
export interface UpdateModelSettingsResponseData {
  success: boolean
  message: string
}

export const updateModelSettingsAPI = async (
  params: ModelConfigProps
): Promise<UpdateModelSettingsResponseData | undefined> => {
  const res = await axiosInstance.post<UpdateModelSettingsResponseData>('/api/model_settings/update', {
    config: params
  })
  return get(res, 'data.data')
}
