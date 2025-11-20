// Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

import { FC, useEffect } from 'react'
import { Form, Button, Select, Input, Typography, Spin, Message } from '@arco-design/web-react'
import { get, isEmpty } from 'lodash'
import { useMemoizedFn, useMount, useRequest } from 'ahooks'

import { ModelTypeList, BaseUrl, ModelInfoList } from './constants'
import { getModelInfo, ModelConfigProps, updateModelSettingsAPI } from '../../services/Settings'

const FormItem = Form.Item
const { Text } = Typography

interface SettingsProps {
  closeSetting?: () => void
  init?: boolean
}

interface InputPrefixProps {
  label: string
}

const InputPrefix: FC<InputPrefixProps> = ({ label }) => {
  return <div className="flex w-[73px] items-center">{label}</div>
}

type SettingsFormProps = Record<string, string>

interface ModelSectionProps {
  prefix: string
  title: string
  description?: string
  optional?: boolean
  onPlatformChange: (prefix: string, platform: ModelTypeList) => void
}

const modelPlatformOptions = ModelInfoList.map((info) => ({
  value: info.value,
  label: (
    <div className="flex items-center gap-2">
      {info.icon}
      <span className="capitalize">{info.key}</span>
    </div>
  )
}))

const defaultBaseUrl: Record<string, string> = {
  [ModelTypeList.Doubao]: BaseUrl.DoubaoUrl,
  [ModelTypeList.OpenAI]: BaseUrl.OpenAIUrl
}

const ModelSection: FC<ModelSectionProps> = ({ prefix, title, description, optional, onPlatformChange }) => {
  const requiredRules = optional ? [] : [{ required: true, message: 'Cannot be empty' }]
  return (
    <div className="flex flex-col gap-[12px] mb-6">
      <div className="flex flex-col gap-1">
        <div className="text-[16px] font-semibold text-[#0B0B0F]">{title}</div>
        {description ? (
          <Text type="secondary" className="text-[12px]">
            {description}
          </Text>
        ) : null}
      </div>
      <FormItem
        label="Model platform"
        field={`${prefix}-modelPlatform`}
        className="!mb-0"
        rules={requiredRules}
        requiredSymbol={false}>
        <Select
          placeholder="Select a provider"
          className="!w-[574px]"
          onChange={(val) => onPlatformChange(prefix, val as ModelTypeList)}>
          {modelPlatformOptions.map((item) => (
            <Select.Option key={item.value} value={item.value}>
              {item.label}
            </Select.Option>
          ))}
        </Select>
      </FormItem>
      <FormItem field={`${prefix}-modelId`} className="!mb-0" rules={requiredRules} requiredSymbol={false}>
        <Input
          addBefore={<InputPrefix label="Model name" />}
          placeholder="Enter model name"
          allowClear
          className="[&_.arco-input-inner-wrapper]: !w-[574px]"
        />
      </FormItem>
      <FormItem field={`${prefix}-baseUrl`} className="!mb-0" rules={requiredRules} requiredSymbol={false}>
        <Input
          addBefore={<InputPrefix label="Base URL" />}
          placeholder="Enter your base URL"
          allowClear
          className="[&_.arco-input-inner-wrapper]: !w-[574px]"
        />
      </FormItem>
      <FormItem field={`${prefix}-apiKey`} className="!mb-0" rules={requiredRules} requiredSymbol={false}>
        <Input
          addBefore={<InputPrefix label="API Key" />}
          placeholder="Enter your API Key"
          allowClear
          className="!w-[574px]"
        />
      </FormItem>
    </div>
  )
}

const Settings: FC<SettingsProps> = ({ closeSetting, init = false }) => {
  const [form] = Form.useForm<SettingsFormProps>()

  const { data: modelInfo, loading: getInfoLoading, run: getInfo } = useRequest(getModelInfo, {
    manual: true
  })

  const { loading: updateLoading, run: updateModelSettings } = useRequest(updateModelSettingsAPI, {
    manual: true,
    onSuccess() {
      Message.success('Your API key saved successfully')
      getInfo()
      if (init) {
        closeSetting?.()
      }
    },
    onError(e: Error) {
      const errMsg = get(e, 'response.data.message') || get(e, 'message') || 'Failed to save settings'
      Message.error(errMsg)
    }
  })

  const handlePlatformChange = useMemoizedFn((prefix: string, platform: ModelTypeList) => {
    if (defaultBaseUrl[platform]) {
      form.setFieldValue(`${prefix}-baseUrl`, defaultBaseUrl[platform])
    }
  })

  const submit = useMemoizedFn(async () => {
    try {
      await form.validate()
      const values = form.getFieldsValue()

      const buildConfig = (key: string, optional?: boolean) => {
        const modelPlatform = values[`${key}-modelPlatform`]
        const modelId = values[`${key}-modelId`]
        const baseUrl = values[`${key}-baseUrl`]
        const apiKey = values[`${key}-apiKey`]
        const hasAny = [modelPlatform, modelId, baseUrl, apiKey].some(Boolean)

        if (optional && !hasAny) return undefined
        if (!modelPlatform || !modelId || !baseUrl || !apiKey) {
          const msg = optional
            ? 'Please complete reranker settings or leave them empty.'
            : 'Required fields cannot be empty.'
          Message.error(msg)
          throw new Error(msg)
        }

        return { modelPlatform, modelId, baseUrl, apiKey, provider: modelPlatform }
      }

      const payload: ModelConfigProps = {
        vlm: buildConfig('vlm')!,
        llm: buildConfig('llm')!,
        embedding: buildConfig('embedding')!
      }

      const rerankerConfig = buildConfig('reranker', true)
      if (rerankerConfig) {
        payload.reranker = rerankerConfig
      }

      updateModelSettings(payload)
    } catch (error) {}
  })

  useMount(() => {
    form.setFieldsValue({
      'vlm-modelPlatform': ModelTypeList.Doubao,
      'vlm-baseUrl': BaseUrl.DoubaoUrl,
      'llm-modelPlatform': ModelTypeList.OpenAI,
      'llm-baseUrl': BaseUrl.OpenAIUrl,
      'embedding-modelPlatform': ModelTypeList.Doubao,
      'embedding-baseUrl': BaseUrl.DoubaoUrl
    })
    getInfo()
  })

  useEffect(() => {
    const config = get(modelInfo, 'config') as ModelConfigProps | undefined
    if (!getInfoLoading && config && !isEmpty(config)) {
      const nextValues: Partial<SettingsFormProps> = {}
      ;(['vlm', 'llm', 'embedding', 'reranker'] as Array<keyof ModelConfigProps>).forEach((key) => {
        const section = config[key]
        if (section) {
          nextValues[`${key}-modelPlatform`] = section.modelPlatform
          nextValues[`${key}-modelId`] = section.modelId
          nextValues[`${key}-baseUrl`] = section.baseUrl
          nextValues[`${key}-apiKey`] = section.apiKey
        }
      })
      form.setFieldsValue(nextValues as SettingsFormProps)
    }
  }, [modelInfo, getInfoLoading])

  return (
    <Spin loading={getInfoLoading} block className="[&_.arco-spin-children]:!h-full !h-full">
      <div className="top-0 left-0 flex flex-col h-full overflow-y-hidden py-2 pr-2 relative">
        <div className="bg-white rounded-[16px] pl-6 flex flex-col h-full overflow-y-auto overflow-x-hidden scrollbar-hide pb-2">
          <div className="mb-[12px]">
            <div className="mt-[26px] mb-[10px] text-[24px] font-bold text-[#000]">Select AI models by task</div>
            <Text type="secondary" className="text-[13px]">
              Configure separate providers for vision, language, embeddings, and optional reranking to balance cost and quality.
            </Text>
          </div>

          <Form autoComplete="off" layout={'vertical'} form={form}>
            <ModelSection
              prefix="vlm"
              title="Vision language model"
              description="Use an economical multimodal model for screen and PDF recognition."
              onPlatformChange={handlePlatformChange}
            />
            <ModelSection
              prefix="llm"
              title="Language model"
              description="High-quality text model for chat, summaries, and planning."
              onPlatformChange={handlePlatformChange}
            />
            <ModelSection
              prefix="embedding"
              title="Embedding model"
              description="Vectorize captured context for retrieval and semantic search."
              onPlatformChange={handlePlatformChange}
            />
            <ModelSection
              prefix="reranker"
              title="Reranker (optional)"
              description="Supply a reranking model to reorder retrieval results; leave empty if unused."
              optional
              onPlatformChange={handlePlatformChange}
            />
          </Form>
          <Spin loading={updateLoading}>
            <Button type="primary" onClick={submit} disabled={updateLoading} className="!bg-[#000]">
              {init ? 'Get started' : 'Save'}
            </Button>
          </Spin>
        </div>
      </div>
    </Spin>
  )
}

export default Settings
