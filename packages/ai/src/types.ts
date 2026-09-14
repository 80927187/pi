import type { TelemetryContext } from "@earendil-works/pi-telemetry";
import type { AnthropicOptions } from "./api/anthropic-messages.ts";
import type { AzureOpenAIResponsesOptions } from "./api/azure-openai-responses.ts";
import type { BedrockOptions } from "./api/bedrock-converse-stream.ts";
import type { GoogleOptions } from "./api/google-generative-ai.ts";
import type { GoogleVertexOptions } from "./api/google-vertex.ts";
import type { MistralOptions } from "./api/mistral-conversations.ts";
import type { OpenAICodexResponsesOptions } from "./api/openai-codex-responses.ts";
import type { OpenAICompletionsOptions } from "./api/openai-completions.ts";
import type { OpenAIResponsesOptions } from "./api/openai-responses.ts";
import type { PiMessagesOptions } from "./api/pi-messages.ts";
import type { AssistantMessageDiagnostic } from "./utils/diagnostics.ts";
import type { AssistantMessageEventStream } from "./utils/event-stream.ts";

export type { AssistantMessageEventStream } from "./utils/event-stream.ts";

export type KnownApi =
	| "openai-completions"
	| "mistral-conversations"
	| "openai-responses"
	| "azure-openai-responses"
	| "openai-codex-responses"
	| "anthropic-messages"
	| "bedrock-converse-stream"
	| "google-generative-ai"
	| "google-vertex"
	| "pi-messages";

export type Api = KnownApi | (string & {});

export type KnownImagesApi = "openrouter-images";

export type ImagesApi = KnownImagesApi | (string & {});

export type KnownProvider =
	| "amazon-bedrock"
	| "ant-ling"
	| "anthropic"
	| "google"
	| "google-vertex"
	| "openai"
	| "azure-openai-responses"
	| "openai-codex"
	| "radius"
	| "nvidia"
	| "deepseek"
	| "github-copilot"
	| "xai"
	| "groq"
	| "cerebras"
	| "openrouter"
	| "vercel-ai-gateway"
	| "zai"
	| "zai-coding-cn"
	| "mistral"
	| "minimax"
	| "minimax-cn"
	| "moonshotai"
	| "moonshotai-cn"
	| "huggingface"
	| "fireworks"
	| "together"
	| "baseten"
	| "opencode"
	| "opencode-go"
	| "kimi-coding"
	| "cloudflare-workers-ai"
	| "cloudflare-ai-gateway"
	| "qwen-token-plan"
	| "qwen-token-plan-cn"
	| "qwen-token-plan-individual"
	| "xiaomi"
	| "xiaomi-token-plan-cn"
	| "xiaomi-token-plan-ams"
	| "xiaomi-token-plan-sgp";
export type ProviderId = KnownProvider | string;

export type KnownImagesProvider = "openrouter";

export type ImagesProviderId = KnownImagesProvider | string;

export type ToolChoice = "auto" | "none";
export type ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
export type ModelThinkingLevel = "off" | ThinkingLevel;
export type ThinkingLevelMap = Partial<Record<ModelThinkingLevel, string | null>>;
export type ChatTemplateKwargValue =
	| string
	| number
	| boolean
	| null
	| {
			$var: "thinking.enabled" | "thinking.effort" | "thinking.budget";
			omitWhenOff?: boolean;
	  };

/** 用于在 OpenAI 兼容服务器上限制推理 token 的顶层请求字段。 */
export type ThinkingTokenBudgetField = "thinking_token_budget" | "thinking_budget" | "thinking_budget_tokens";

/** 各思考层级的 token 预算（仅适用于基于 token 的 provider） */
export interface ThinkingBudgets {
	minimal?: number;
	low?: number;
	medium?: number;
	high?: number;
}

// 所有 provider 共享的基础选项
export type CacheRetention = "none" | "short" | "long";

export type Transport = "sse" | "websocket" | "websocket-cached" | "auto";

/** provider 作用域的环境变量覆盖。其值优先于 process.env。 */
export type ProviderEnv = Record<string, string>;
export type ProviderHeaders = Record<string, string | null>;
export type FetchFunction = typeof globalThis.fetch;
export type SessionAffinityFormat = "openai" | "openai-nosession" | "openrouter";

export interface ProviderResponse {
	status: number;
	headers: Record<string, string>;
}

/** provider 请求共享的认证、HTTP 传输与生命周期回调。 */
export interface ProviderRequestOptions<TModel = Model<Api>> {
	signal?: AbortSignal;
	/** 此逻辑请求所产生的遥测数据的显式父上下文。 */
	telemetryContext?: TelemetryContext;
	apiKey?: string;
	/**
	 * 用于 provider HTTP 请求的可选 fetch 实现。
	 * 默认为 `globalThis.fetch`。无法注入自定义实现的 provider 适配器可能会拒绝它。
	 * 这不会影响 WebSocket 传输。
	 */
	fetch?: FetchFunction;
	/**
	 * provider 作用域的环境变量值。对于区域设置、endpoint 占位符和代理变量等
	 * provider 配置，这些值优先于 process.env。
	 */
	env?: ProviderEnv;
	/**
	 * 可选回调，用于在发送前检查或替换 provider 载荷。
	 * 返回 undefined 表示保持载荷不变。
	 */
	onPayload?: (payload: unknown, model: TModel) => unknown | undefined | Promise<unknown | undefined>;
	/**
	 * 在收到 HTTP 响应后调用的可选回调。
	 */
	onResponse?: (response: ProviderResponse, model: TModel) => void | Promise<void>;
	/**
	 * 包含在 API 请求中的可选自定义 HTTP 头。
	 * 与 provider 默认值合并；调用方提供的值会覆盖默认头。
	 * 在 AWS Bedrock 上，这些头通过 Smithy `build` 阶段中间件注入，因此
	 * 会被 SigV4 签名覆盖；为保留 SigV4 / bearer 认证，保留头（`x-amz-*`、
	 * `authorization`、`host`）会被静默忽略。
	 * 值为 null 会抑制同名的 provider/API 默认头。
	 */
	headers?: ProviderHeaders;
	/**
	 * 支持超时设置的 provider/SDK 的 HTTP 请求超时（毫秒）。
	 * 例如，OpenAI 和 Anthropic SDK 客户端默认 10 分钟。
	 */
	timeoutMs?: number;
	/**
	 * 支持客户端重试的 provider/SDK 的最大重试次数。
	 * 例如，OpenAI 和 Anthropic SDK 客户端默认为 2。
	 */
	maxRetries?: number;
	/**
	 * 当服务器要求较长等待时，重试所等待的最大延迟（毫秒）。
	 * 如果服务器请求的延迟超过此值，请求会立即失败，并在错误中包含所请求的延迟，
	 * 交由上层重试逻辑在用户可见的情况下处理。
	 * 默认：60000（60 秒）。设为 0 可禁用该上限。
	 */
	maxRetryDelayMs?: number;
}

export interface StreamOptions extends ProviderRequestOptions<Model<Api>> {
	/**
	 * 可选回调，在收到 HTTP 响应之后、其响应体流被消费之前调用。
	 */
	onResponse?: (response: ProviderResponse, model: Model<Api>) => void | Promise<void>;
	temperature?: number;
	/**
	 * 任意采样参数，按原样合并进请求体（位于具名请求字段之后），因此此处的键会覆盖它们。
	 * 让自定义的 OpenAI 兼容服务器（llama.cpp、vLLM、SGLang 等）能接收 pi 未建模的参数，
	 * 例如 `top_p`、`top_k`、`min_p`、`repetition_penalty`。按每个键覆盖 `Model.samplingParams`。
	 * 仅由 OpenAI 兼容适配器（completions、responses、Azure responses）应用；其他 API 会忽略它。
	 */
	samplingParams?: Record<string, unknown>;
	maxTokens?: number;
	/**
	 * 支持多种传输的 provider 的首选传输方式。
	 * 不支持此选项的 provider 会忽略它。
	 */
	transport?: Transport;
	/**
	 * 提示缓存保留偏好。provider 会将其映射到自己支持的值。
	 * 默认："short"。
	 */
	cacheRetention?: CacheRetention;
	/**
	 * 用于支持基于会话缓存的 provider 的可选会话标识符。
	 * provider 可用它来启用提示缓存、请求路由或其他会话感知特性。
	 * 不支持的 provider 会忽略它。
	 */
	sessionId?: string;
	/**
	 * 支持 WebSocket 传输的 provider 的 WebSocket 连接超时（毫秒）。
	 * 这仅涵盖连接/打开握手；连接后流的空闲检测使用 timeoutMs。
	 */
	websocketConnectTimeoutMs?: number;
	/**
	 * 包含在 API 请求中的可选元数据。
	 * provider 会提取其理解的字段并忽略其余字段。
	 * 例如，Anthropic 使用 `user_id` 进行滥用追踪和限流。
	 */
	metadata?: Record<string, unknown>;
}

export type ProviderStreamOptions = StreamOptions & Record<string, unknown>;

export interface DeferredFetchOptions extends ProviderRequestOptions<Model<Api>> {
	/**
	 * provider 长轮询的最长持续时间（毫秒）。
	 * 默认为 0，即仅执行一次状态检查。
	 */
	wait?: number;
}

/** 尽力而为的延迟响应取消操作的请求选项。 */
export type DeferredCancelOptions = ProviderRequestOptions<Model<Api>>;

/**
 * 将已知 API 映射到其完整的 provider 特定流选项类型。
 * 来自 API 实现模块的纯类型导入在产物生成时会被擦除，
 * 因此这不会影响 tree-shaking。
 */
export interface ApiOptionsMap {
	"anthropic-messages": AnthropicOptions;
	"openai-completions": OpenAICompletionsOptions;
	"openai-responses": OpenAIResponsesOptions;
	"openai-codex-responses": OpenAICodexResponsesOptions;
	"azure-openai-responses": AzureOpenAIResponsesOptions;
	"google-generative-ai": GoogleOptions;
	"google-vertex": GoogleVertexOptions;
	"mistral-conversations": MistralOptions;
	"bedrock-converse-stream": BedrockOptions;
	"pi-messages": PiMessagesOptions;
}

/**
 * 某个 API 的完整流选项。已知 API 解析为其具体选项类型；
 * 自定义 API 字符串则回退到通用结构。
 */
export type ApiStreamOptions<TApi extends Api> = TApi extends keyof ApiOptionsMap
	? ApiOptionsMap[TApi]
	: StreamOptions & Record<string, unknown>;

/**
 * API 实现模块统一遵循的流契约：`src/api/` 下的每个模块都导出 `stream` 和
 * `streamSimple`；具备能力的模块还可导出延迟响应方法。懒加载包装器（`lazyApi()`）
 * 和 provider 工厂将其作为值传递。这是无类型的调度结构；
 * 各 API 的选项类型定义在实现模块自身，以及通过 `ApiStreamOptions` 定义在
 * `Provider.stream()` 上。
 */
export interface ProviderStreams {
	stream(model: Model<Api>, context: Context, options?: StreamOptions): AssistantMessageEventStream;
	streamSimple(model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream;
	fetchDeferred?(
		model: Model<Api>,
		handle: DeferredHandle,
		options?: DeferredFetchOptions,
	): AssistantMessageEventStream;
	cancelDeferred?(model: Model<Api>, handle: DeferredHandle, options?: DeferredCancelOptions): Promise<void>;
}

/**
 * 图像生成 API 实现模块统一遵循的契约：`src/api/` 下的每个图像 API 模块都恰好
 * 导出 `generateImages`，因此模块本身即满足此接口。懒加载包装器和图像
 * provider 工厂将其作为值传递。
 */
export interface ProviderImages {
	generateImages(
		model: ImagesModel<ImagesApi>,
		context: ImagesContext,
		options?: ImagesOptions,
	): Promise<AssistantImages>;
}

export interface ImagesOptions extends ProviderRequestOptions<ImagesModel<ImagesApi>> {
	/**
	 * 包含在 API 请求中的可选元数据。
	 * provider 会提取其理解的字段并忽略其余字段。
	 */
	metadata?: Record<string, unknown>;
}

export type ProviderImagesOptions = ImagesOptions & Record<string, unknown>;

export interface AnthropicAllowedFallbackModel {
	provider: ProviderId;
	model: string;
	cost: ModelCost;
}

// 传递给 streamSimple() 和 completeSimple() 的含推理设置的统一选项
export interface SimpleStreamOptions extends StreamOptions {
	/** 简单请求的 provider 中立工具选择。省略时，适配器使用 provider 特定的行为。 */
	toolChoice?: ToolChoice;
	reasoning?: ThinkingLevel;
	/** 要求具备能力的 provider 返回持久句柄并异步继续请求。 */
	deferred?: boolean | { window?: "15m" | "1h" | "24h" };
	/** 各思考层级的自定义 token 预算（仅适用于基于 token 的 provider） */
	thinkingBudgets?: ThinkingBudgets;
}

// 带类型化选项的通用 StreamFunction。
//
// 契约：
// - 必须返回一个 AssistantMessageEventStream。
// - 当请求认证缺失时，直接调用 streamSimple() 可能会同步抛出异常。
//   一旦返回了流，请求/模型/运行时失败应被编码到该流中。
// - 错误终止必须生成一个 stopReason 为 "error" 或 "aborted" 且带有
//   errorMessage 的 AssistantMessage，并通过流协议发出。
export type StreamFunction<TApi extends Api = Api, TOptions extends StreamOptions = StreamOptions> = (
	model: Model<TApi>,
	context: Context,
	options?: TOptions,
) => AssistantMessageEventStream;

export type ImagesFunction<TApi extends ImagesApi = ImagesApi, TOptions extends ImagesOptions = ImagesOptions> = (
	model: ImagesModel<TApi>,
	context: ImagesContext,
	options?: TOptions,
) => Promise<AssistantImages>;

export interface TextSignatureV1 {
	v: 1;
	id: string;
	phase?: "commentary" | "final_answer";
}

export interface TextContent {
	type: "text";
	text: string;
	textSignature?: string; // 例如，对于 OpenAI responses，为消息元数据（旧版 id 字符串或 TextSignatureV1 JSON）
}

export interface ThinkingContent {
	type: "thinking";
	thinking: string;
	thinkingSignature?: string; // provider 特定的不透明或序列化的推理重放数据
	/** 为 true 时，思考内容被安全过滤器抹除。不透明的加密载荷存储在
	 *  `thinkingSignature` 中，以便回传给 API 以保持多轮连续性。 */
	redacted?: boolean;
}

export interface ImageContent {
	type: "image";
	data: string; // base64 编码的图像数据
	mimeType: string; // 例如 "image/jpeg"、"image/png"
}

export interface ToolCall {
	type: "toolCall";
	id: string;
	name: string;
	arguments: Record<string, any>;
	thoughtSignature?: string; // Google 特有：用于复用思考上下文的不透明签名
	/** OpenAI Responses 中用于调用动态加载或带命名空间工具的名称空间。 */
	namespace?: string;
}

export interface Usage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	/** `cacheWrite` 中以 1h 保留策略写入的子集。仅 Anthropic 报告此拆分。 */
	cacheWrite1h?: number;
	/**
	 * provider 报告时的推理/思考 token 数。这是 `output` 的子集：
	 * `output` 已包含这些 token。提供推理明细的 provider 会设为数值（可能为 0）；
	 * 不提供的 provider 则保留为 undefined。
	 */
	reasoning?: number;
	totalTokens: number;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total: number;
	};
}

export type StopReason = "pending" | "stop" | "length" | "toolUse" | "error" | "aborted" | "deferred";

export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

export interface DeferredHandle {
	provider: string;
	modelId: string;
	api: string;
	/** provider token，例如响应 id 或批处理 id 加行 id。 */
	id: string;
	expiresAt?: number;
	pollAfterMs?: number;
	/** 重建最终 assistant 消息所需的 provider 转换数据。 */
	data?: JsonValue;
}

export interface UserMessage {
	role: "user";
	content: string | (TextContent | ImageContent)[];
	timestamp: number; // Unix 时间戳（毫秒）
}

export interface AssistantMessage {
	role: "assistant";
	content: (TextContent | ThinkingContent | ToolCall)[];
	api: Api;
	provider: ProviderId;
	model: string;
	responseModel?: string; // 当与实际请求的 `model` 不同时，为具体的 `chunk.model`（例如 OpenRouter 的 `auto` -> `anthropic/...`）
	responseId?: string; // 上游 API 提供时，provider 特定的响应/消息标识符
	/** 此响应所使用的 provider 原生 effort 级别。旧版或非托管响应没有该字段。 */
	providerThinkingLevel?: string;
	diagnostics?: AssistantMessageDiagnostic[]; // 用于失败与恢复的、已脱敏的 provider/运行时诊断信息。
	usage: Usage;
	stopReason: StopReason;
	deferred?: DeferredHandle;
	errorMessage?: string;
	rawStopReason?: string;
	/**
	 * provider 关于模型是否显式结束其轮次的指示。
	 * 为调试而保留，目前不影响 agent 控制流。
	 */
	endTurn?: boolean;
	timestamp: number; // Unix 时间戳（毫秒）
}

export interface ToolResultMessage<TDetails = any> {
	role: "toolResult";
	toolCallId: string;
	toolName: string;
	content: (TextContent | ImageContent)[]; // 支持文本和图像
	details?: TDetails;
	/** 来自工具执行本身的用量（若有）。不属于主 LLM 上下文核算的一部分。 */
	usage?: Usage;
	/**
	 * 在此结果之后变为可用的 `Context.tools` 中的名称。
	 * 具有原生延迟工具加载能力的 provider 将其作为加载点；
	 * 其他 provider 会忽略它，按常规使用 `Context.tools`。
	 */
	addedToolNames?: string[];
	isError: boolean;
	timestamp: number; // Unix 时间戳（毫秒）
}

export type Message = UserMessage | AssistantMessage | ToolResultMessage;

export type ImagesInputContent = TextContent | ImageContent;
export type ImagesOutputContent = TextContent | ImageContent;

export interface ImagesContext {
	input: ImagesInputContent[];
}

export type ImagesStopReason = "stop" | "error" | "aborted";

export interface AssistantImages {
	api: ImagesApi;
	provider: ImagesProviderId;
	model: string;
	output: ImagesOutputContent[];
	responseId?: string;
	usage?: Usage;
	stopReason: ImagesStopReason;
	errorMessage?: string;
	timestamp: number; // Unix 时间戳（毫秒）
}

import type { TSchema } from "typebox";

/** 用于约束采样的 OpenAI 语法变体。 */
export type GrammarFormat = "openai_lark" | "openai_regex";

export type GrammarVariants = Partial<Record<GrammarFormat, string>>;

/**
 * 工具的可选 provider 端约束采样配置。
 *
 * `json_schema` 值大致对应 API 中 `strict` 的概念，由 API 实现为 json-schema
 * 约束采样。语法变体允许调用方为同一目标语言提供 provider 特定的编码。
 */
export type ConstrainedSamplingConfig =
	| {
			type: "json_schema";
			strict: "prefer" | "require";
	  }
	| {
			type: "grammar";
			variants: GrammarVariants;
	  };

export interface Tool<TParameters extends TSchema = TSchema> {
	name: string;
	description: string;
	parameters: TParameters;
	constrainedSampling?: false | ConstrainedSamplingConfig;
}

export interface Context {
	systemPrompt?: string;
	messages: Message[];
	tools?: Tool[];
}

/**
 * AssistantMessageEventStream 的事件协议。
 *
 * 成功的流会在部分更新之前发出 `start`，并以 `done` 终止。当请求在生成开始前
 * 设置失败时，流可能直接以 `error` 终止；在 `start` 之后，失败也以 `error` 终止。
 * 当请求认证缺失时，直接调用 `streamSimple()` 会同步抛出异常。
 * 更新和 `done` 绝不能出现在 `start` 之前。
 *
 * `partial` 是共享的“截至目前实时响应”辅助对象，而非事件时刻的快照。
 * 文本和思考块在其 `*_start` 事件发出时为空，仅通过对应的 `*_delta` 事件增长，
 * 直到权威的 `*_end`。被抹除的思考可能在 start 时即完整，且不发出任何 delta。
 * `toolcall_start` 处的工具调用参数是 provider 特定的；`toolcall_delta` 携带
 * 后续的 JSON 更新。
 */
export type AssistantMessageEvent =
	| { type: "start"; partial: AssistantMessage }
	| { type: "text_start"; contentIndex: number; partial: AssistantMessage }
	| { type: "text_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
	| { type: "text_end"; contentIndex: number; content: string; partial: AssistantMessage }
	| { type: "thinking_start"; contentIndex: number; partial: AssistantMessage }
	| { type: "thinking_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
	| { type: "thinking_end"; contentIndex: number; content: string; partial: AssistantMessage }
	| { type: "toolcall_start"; contentIndex: number; partial: AssistantMessage }
	| { type: "toolcall_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
	| { type: "toolcall_end"; contentIndex: number; toolCall: ToolCall; partial: AssistantMessage }
	| {
			type: "done";
			reason: Extract<StopReason, "stop" | "length" | "toolUse" | "deferred">;
			message: AssistantMessage;
	  }
	| { type: "error"; reason: Extract<StopReason, "aborted" | "error">; error: AssistantMessage };

/**
 * OpenAI 兼容 completions API 的兼容性设置。
 * 用于为自定义 provider 覆盖基于 URL 的自动检测。
 */
export interface OpenAICompletionsCompat {
	/** provider 是否支持 `store` 字段。默认：根据 URL 自动检测。 */
	supportsStore?: boolean;
	/** provider 是否支持 `developer` 角色（相对于 `system`）。默认：根据 URL 自动检测。 */
	supportsDeveloperRole?: boolean;
	/** provider 是否支持 `reasoning_effort`。默认：根据 URL 自动检测。 */
	supportsReasoningEffort?: boolean;
	/** provider 是否支持 `stream_options: { include_usage: true }` 以在流式响应中返回 token 用量。默认：true。 */
	supportsUsageInStreaming?: boolean;
	/** 流式响应是否包含 `finish_reason`。为 false 时，pi 在流结束时推断为 `stop` 或 `toolUse`。默认：true。 */
	supportsFinishReason?: boolean;
	/** 使用哪个字段表示 max tokens。默认：根据 URL 自动检测。 */
	maxTokensField?: "max_completion_tokens" | "max_tokens";
	/** 工具结果是否要求 `name` 字段。默认：根据 URL 自动检测。 */
	requiresToolResultName?: boolean;
	/** 工具结果之后的 user 消息是否需要中间插入一条 assistant 消息。默认：根据 URL 自动检测。 */
	requiresAssistantAfterToolResult?: boolean;
	/** 思考块是否必须转换为带 <thinking> 分隔符的文本块。默认：根据 URL 自动检测。 */
	requiresThinkingAsText?: boolean;
	/** 启用推理时，所有重放的 assistant 消息是否必须包含空的 reasoning_content 字段。默认：根据 URL 自动检测。 */
	requiresReasoningContentOnAssistantMessages?: boolean;
	/** 推理/思考参数的格式。"openai" 使用 reasoning_effort，"openrouter" 使用 reasoning: { effort }，"deepseek" 使用 thinking: { type }，并在支持时附加 reasoning_effort，"together" 使用 reasoning: { enabled }，并在支持时附加 reasoning_effort，"baseten" 使用可配置的 chat_template_args，并在支持时附加 reasoning_effort，"zai" 使用 thinking: { type }，"qwen" 使用顶层 enable_thinking: boolean，"qwen-chat-template" 使用 chat_template_kwargs.enable_thinking 和 preserve_thinking，"chat-template" 使用可配置的 chat_template_kwargs，"string-thinking" 使用顶层 thinking: string，"ant-ling" 仅当映射后的 effort 非 null 时使用 reasoning: { effort }。默认："openai"。 */
	thinkingFormat?:
		| "openai"
		| "openrouter"
		| "deepseek"
		| "together"
		| "baseten"
		| "zai"
		| "qwen"
		| "chat-template"
		| "qwen-chat-template"
		| "string-thinking"
		| "ant-ling";
	/** 当 `thinkingFormat` 为 `chat-template` 时，作为 `chat_template_kwargs` 发送的 Kwargs。使用 `{ "$var": "thinking.enabled" }`、`{ "$var": "thinking.effort" }` 或 `{ "$var": "thinking.budget" }` 表示由 pi 控制的思考值。 */
	chatTemplateKwargs?: Record<string, ChatTemplateKwargValue>;
	/** 当 `thinkingFormat` 为 `baseten` 时，作为 `chat_template_args` 发送的参数。使用 `{ "$var": "thinking.enabled" }`、`{ "$var": "thinking.effort" }` 或 `{ "$var": "thinking.budget" }` 表示由 pi 控制的思考值。 */
	chatTemplateArgs?: Record<string, ChatTemplateKwargValue>;
	/** 作为 `provider` 请求字段发送的 OpenRouter 兼容路由偏好。 */
	openRouterRouting?: OpenRouterRouting;
	/** Vercel AI Gateway 路由偏好。仅在 baseUrl 指向 Vercel AI Gateway 时使用。 */
	vercelGatewayRouting?: VercelGatewayRouting;
	/** z.ai 是否支持顶层 `tool_stream: true` 以流式返回工具调用 delta。默认：false。 */
	zaiToolStream?: boolean;
	/**
	 * 用于限制 `thinkingBudgets` 推理 token 的顶层请求字段。
	 * 在这些 endpoint 上，推理和答案共享 `max_tokens`，因此若无预算，
	 * 推理密集的轮次可能耗尽整个响应而不输出答案。
	 * `"thinking_token_budget"` 适用于 vLLM，`"thinking_budget"` 适用于 Qwen/DashScope/SGLang，
	 * `"thinking_budget_tokens"` 适用于 llama.cpp。默认关闭；不在生成的目录中设置。
	 */
	thinkingTokenBudgetField?: ThinkingTokenBudgetField;
	/** `thinkingTokenBudgetField: "thinking_token_budget"`（vLLM）的别名。优先使用 `thinkingTokenBudgetField`。默认：false。 */
	supportsThinkingTokenBudget?: boolean;
	/** provider 是否支持带 Lark/regex 语法格式的 OpenAI 自定义工具。为 false 时，语法约束工具回退为普通函数工具。默认：false；生成的模型目录会为具备能力的模型启用它。 */
	supportsOpenAIGrammarTools?: boolean;
	/** provider 是否支持工具定义中的 `strict` 字段。默认：true。 */
	supportsStrictMode?: boolean;
	/** 提示缓存的缓存控制约定。"anthropic" 会将 Anthropic 风格的 `cache_control` 标记应用于系统提示、最后一个工具定义，以及最后一条 user、assistant 或 tool-result 的文本内容。 */
	cacheControlFormat?: "anthropic";
	/** 是否发送来自 `options.sessionId` 的会话亲和性数据。默认：OpenRouter endpoint 为 true，其他为 false。 */
	sendSessionAffinityHeaders?: boolean;
	/** provider 特定的延迟工具序列化模式。 */
	deferredToolsMode?: "kimi";
	/** 会话亲和性头格式：`openai` 发送 `session_id`、`x-client-request-id` 和 `x-session-affinity`；`openai-nosession` 发送 `x-client-request-id` 和 `x-session-affinity`；`openrouter` 发送 `x-session-id`。不影响 `prompt_cache_key` 请求体参数，后者由缓存保留策略控制。默认：自动检测。 */
	sessionAffinityFormat?: SessionAffinityFormat;
	/** provider 是否支持长时间提示缓存保留（`prompt_cache_retention: "24h"` 或 Anthropic 风格的 `cache_control.ttl: "1h"`，取决于格式）。默认：true。 */
	supportsLongCacheRetention?: boolean;
	/**
	 * 作为顶层 `priority` 请求字段发送的 vLLM 调度器优先级（值越小越早处理；
	 * 服务器默认 0）。仅当 vLLM 使用 `--scheduling-policy priority` 运行时才有意义；
	 * 可用于避免后台/批处理工作拖慢交互式会话。默认关闭；不在生成的目录中设置。
	 */
	vllmPriority?: number;
}

/** OpenAI Responses API 的兼容性设置。 */
export interface OpenAIResponsesCompat {
	/** provider 是否支持 `developer` 角色（相对于 `system`）。默认：true。 */
	supportsDeveloperRole?: boolean;
	/** 会话亲和性头格式：`openai` 发送 `session_id` 和 `x-client-request-id`；`openai-nosession` 发送 `x-client-request-id`；`openrouter` 发送 `x-session-id`。不影响 `prompt_cache_key` 请求体参数，后者由缓存保留策略控制。默认：自动检测。 */
	sessionAffinityFormat?: SessionAffinityFormat;
	/** provider 是否支持长时间提示缓存保留。在 GPT-5.6+ 上使用 `prompt_cache_options.ttl: "30m"`，在更早的模型上使用 `prompt_cache_retention: "24h"`。默认：true。 */
	supportsLongCacheRetention?: boolean;
	/** provider 是否支持严格的 JSON-schema 函数工具。默认值因 API 而异；生成的 OpenAI 模型会显式启用。 */
	supportsStrictMode?: boolean;
	/** 是否生成带 Lark/regex 语法格式的 OpenAI 自定义工具。为 false 时，语法约束工具回退为普通函数工具。默认：false；生成的模型目录会为具备能力的模型启用它。 */
	supportsOpenAIGrammarTools?: boolean;
	/** 模型是否支持以消息为锚点的 `additional_tools` 输入项。默认：false。 */
	supportsAdditionalTools?: boolean;
	/** 模型是否支持对延迟工具执行客户端工具搜索。默认：false。 */
	supportsToolSearch?: boolean;
	/** 模型是否接受 `prompt_cache_options`（OpenAI GPT-5.6+ 提示缓存）。较旧的 OpenAI 模型会拒绝该参数。默认：false。 */
	supportsExplicitPromptCacheMode?: boolean;
	/** provider 是否接受 `max_output_tokens` 参数。某些 Codex 协议网关会拒绝它。默认：true。 */
	supportsMaxOutputTokens?: boolean;
}

/** Anthropic Messages 兼容 API 的兼容性设置。 */
export interface AnthropicMessagesCompat {
	/**
	 * provider 是否接受每个工具的 `eager_input_streaming`。
	 * 为 false 时，Anthropic provider 会省略 `tools[].eager_input_streaming`，
	 * 并对启用工具的请求发送旧版 `fine-grained-tool-streaming-2025-05-14` beta 头。
	 * 默认：true。
	 */
	supportsEagerToolInputStreaming?: boolean;
	/** provider 是否支持 Anthropic 长时间缓存保留（`cache_control.ttl: "1h"`）。默认：true。 */
	supportsLongCacheRetention?: boolean;
	/**
	 * 启用缓存时，是否发送来自 `options.sessionId` 的 `x-session-affinity` 头。
	 * 对于 Fireworks 等使用会话亲和性进行提示缓存路由的 provider 是必需的
	 * （发往同一副本的请求可最大化缓存命中）。
	 * 默认：false。
	 */
	sendSessionAffinityHeaders?: boolean;
	/** 会话亲和性格式。`"openrouter"` 发送 `x-session-id`；未设置时发送 `x-session-affinity`。 */
	sessionAffinityFormat?: "openrouter";
	/**
	 * provider 是否支持在工具定义上使用 Anthropic 风格的 `cache_control` 标记。
	 * 为 false 时，工具参数中会省略 `cache_control`。
	 * 某些 Anthropic 兼容 provider（如 Fireworks）不支持工具上的该字段，
	 * 可能会拒绝或忽略它。
	 * 默认：true。
	 */
	supportsCacheControlOnTools?: boolean;
	/**
	 * 模型是否接受 Anthropic 的 `temperature` 请求字段。
	 * Claude Opus 4.7+ 会拒绝非默认的 temperature 值。
	 * 默认：true。
	 */
	supportsTemperature?: boolean;
	/**
	 * 是否强制使用自适应思考（`thinking.type: "adaptive"` 加上
	 * `output_config.effort`），与模型 id 无关。需要自适应思考的内置模型会在
	 * 生成的元数据中设置此项。自定义的 Anthropic 兼容 provider 可为任何上游要求
	 * 自适应格式的模型将其设为 `true`。设为 `false` 可在被覆盖的内置模型上
	 * 选择退出。
	 * 默认：false。
	 */
	forceAdaptiveThinking?: boolean;
	/** 是否将空的思考签名重放为 `signature: ""`，而不是把思考转换为文本。默认：false。 */
	allowEmptySignature?: boolean;
	/** provider 是否支持 Anthropic 严格工具 schema。默认：false；生成的 Anthropic 模型会显式启用。 */
	supportsStrictTools?: boolean;
	/** 具体的模型传输是否支持仅含 effort 的系统消息和思考绑定控制。默认：false。 */
	supportsMidConvoEffort?: boolean;
	/**
	 * Anthropic 在 `fallbacks` 中接受的模型，用于服务端拒答回退，
	 * 并附带返回的回退响应的本地定价元数据。当缺失或为空时，
	 * 调用方必须省略 `fallbacks`；对于没有允许回退目标的模型，Anthropic 会拒绝该字段。
	 */
	allowedFallbackModels?: AnthropicAllowedFallbackModel[];
	/**
	 * provider 是否支持由工具结果中的 `tool_reference` 块加载的延迟工具。
	 * 默认：对第一方 Anthropic 模型为 true，但 Haiku 和早于 Claude 4.5 的模型除外；
	 * 对其他 provider 为 false。
	 */
	supportsToolReferences?: boolean;
}

/** Amazon Bedrock 模型的兼容性设置。 */
export interface BedrockCompat {
	/** 模型是否支持 Bedrock 严格工具 schema。默认：false。 */
	supportsStrictMode?: boolean;
}

/**
 * OpenRouter provider 路由偏好。
 * 控制 OpenRouter 将请求路由到哪些上游 provider。
 * 作为 OpenRouter API 请求体中的 `provider` 字段发送。
 * @see https://openrouter.ai/docs/guides/routing/provider-selection
 */
export interface OpenRouterRouting {
	/** 是否允许备用 provider 处理请求。默认：true。 */
	allow_fallbacks?: boolean;
	/** 是否仅保留支持请求中所有参数的 provider。默认：false。 */
	require_parameters?: boolean;
	/** 数据收集设置。"allow"（默认）：允许可能存储/训练数据的 provider。"deny"：仅使用不收集用户数据的 provider。 */
	data_collection?: "deny" | "allow";
	/** 是否将路由限制为仅使用 ZDR（零数据保留）endpoint。 */
	zdr?: boolean;
	/** 是否将路由限制为仅使用允许文本蒸馏的模型。 */
	enforce_distillable_text?: boolean;
	/** 按顺序尝试的 provider 名称/slug 有序列表，不可用时回退到下一个。 */
	order?: string[];
	/** 此请求中仅允许使用的 provider 名称/slug 列表。 */
	only?: string[];
	/** 此请求中要跳过的 provider 名称/slug 列表。 */
	ignore?: string[];
	/** 用于筛选 provider 的量化级别列表（例如 ["fp16", "bf16", "fp8", "fp6", "int8", "int4", "fp4", "fp32"]）。 */
	quantizations?: string[];
	/** 排序策略。可以是字符串（例如 "price"、"throughput"、"latency"），也可以是包含 `by` 和 `partition` 的对象。 */
	sort?:
		| string
		| {
				/** 排序指标："price"、"throughput"、"latency"。 */
				by?: string;
				/** 分区策略："model"（默认）或 "none"。 */
				partition?: string | null;
		  };
	/** 每百万 token 的最高价格（USD）。 */
	max_price?: {
		/** 每百万 prompt token 的价格。 */
		prompt?: number | string;
		/** 每百万 completion token 的价格。 */
		completion?: number | string;
		/** 每张图像的价格。 */
		image?: number | string;
		/** 每音频单元的价格。 */
		audio?: number | string;
		/** 每次请求的价格。 */
		request?: number | string;
	};
	/** 首选最低吞吐量（tokens/秒）。可以是数值（应用于 p50）或包含各百分位阈值的对象。 */
	preferred_min_throughput?:
		| number
		| {
				/** 第 50 百分位的最小 tokens/秒。 */
				p50?: number;
				/** 第 75 百分位的最小 tokens/秒。 */
				p75?: number;
				/** 第 90 百分位的最小 tokens/秒。 */
				p90?: number;
				/** 第 99 百分位的最小 tokens/秒。 */
				p99?: number;
		  };
	/** 首选最大延迟（秒）。可以是数值（应用于 p50）或包含各百分位阈值的对象。 */
	preferred_max_latency?:
		| number
		| {
				/** 第 50 百分位的最大延迟（秒）。 */
				p50?: number;
				/** 第 75 百分位的最大延迟（秒）。 */
				p75?: number;
				/** 第 90 百分位的最大延迟（秒）。 */
				p90?: number;
				/** 第 99 百分位的最大延迟（秒）。 */
				p99?: number;
		  };
}

/**
 * Vercel AI Gateway 路由偏好。
 * 控制网关将请求路由到哪些上游 provider。
 * @see https://vercel.com/docs/ai-gateway/models-and-providers/provider-options
 */
export interface VercelGatewayRouting {
	/** 此请求中仅使用的 provider slug 列表（例如 ["bedrock", "anthropic"]）。 */
	only?: string[];
	/** 按顺序尝试的 provider slug 列表（例如 ["anthropic", "openai"]）。 */
	order?: string[];
}

export interface ModelCostRates {
	input: number; // $/百万 tokens
	output: number; // $/百万 tokens
	cacheRead: number; // $/百万 tokens
	cacheWrite: number; // $/百万 tokens
}

export interface ModelCostTier extends ModelCostRates {
	/** 当请求的总输入用量超过此 token 数时使用该档位。 */
	inputTokensAbove: number;
}

export interface ModelCost extends ModelCostRates {
	/** 请求级定价档位。匹配到的最高输入阈值将应用于整个请求。 */
	tiers?: ModelCostTier[];
}

// 统一模型系统的 Model 接口
export interface Model<TApi extends Api> {
	id: string;
	name: string;
	api: TApi;
	provider: ProviderId;
	baseUrl: string;
	reasoning: boolean;
	/**
	 * 将 pi 的思考层级映射到 provider/模型特定的值。
	 * 缺失的键使用 provider 默认值。null 表示该层级不受支持。
	 */
	thinkingLevelMap?: ThinkingLevelMap;
	input: ("text" | "image")[];
	cost: ModelCost;
	contextWindow: number;
	maxTokens: number;
	/** 此模型的默认采样参数。参见 {@link StreamOptions.samplingParams}；每个请求的键会覆盖这些值。 */
	samplingParams?: Record<string, unknown>;
	headers?: Record<string, string>;
	/** OpenAI 兼容 API 的兼容性覆盖。未设置时，根据 baseUrl 自动检测。 */
	compat?: TApi extends "openai-completions"
		? OpenAICompletionsCompat
		: TApi extends "openai-responses" | "azure-openai-responses" | "openai-codex-responses"
			? OpenAIResponsesCompat
			: TApi extends "anthropic-messages"
				? AnthropicMessagesCompat
				: TApi extends "bedrock-converse-stream"
					? BedrockCompat
					: never;
}

export interface ImagesModel<TApi extends ImagesApi>
	extends Omit<Model<Api>, "api" | "provider" | "reasoning" | "contextWindow" | "maxTokens" | "compat"> {
	api: TApi;
	provider: ImagesProviderId;
	output: ("text" | "image")[];
}
