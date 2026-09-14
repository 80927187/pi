import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { type ImageContent, type Message, type TextContent, type Usage, uuidv7 } from "@earendil-works/pi-ai";
import { randomUUID } from "crypto";
import {
	appendFileSync,
	closeSync,
	createReadStream,
	existsSync,
	mkdirSync,
	openSync,
	readdirSync,
	readSync,
	statSync,
	writeFileSync,
} from "fs";
import { readdir, stat } from "fs/promises";
import { join, resolve } from "path";
import { createInterface } from "readline";
import { StringDecoder } from "string_decoder";
import { APP_NAME, getAgentDir as getDefaultAgentDir, getSessionsDir } from "../config.ts";
import { normalizePath, resolvePath } from "../utils/paths.ts";
import {
	type BashExecutionMessage,
	type CustomMessage,
	createBranchSummaryMessage,
	createCompactionSummaryMessage,
	createCustomMessage,
} from "./messages.ts";

export const CURRENT_SESSION_VERSION = 3;

export interface SessionHeader {
	type: "session";
	version?: number; // v1 会话没有该字段
	id: string;
	timestamp: string;
	cwd: string;
	parentSession?: string;
}

export interface NewSessionOptions {
	id?: string;
	parentSession?: string;
}

export interface SessionEntryBase {
	type: string;
	id: string;
	parentId: string | null;
	timestamp: string;
}

export interface SessionMessageEntry extends SessionEntryBase {
	type: "message";
	message: AgentMessage;
}

export interface ThinkingLevelChangeEntry extends SessionEntryBase {
	type: "thinking_level_change";
	thinkingLevel: string;
}

export interface ModelChangeEntry extends SessionEntryBase {
	type: "model_change";
	provider: string;
	modelId: string;
}

export interface CompactionEntry<T = unknown> extends SessionEntryBase {
	type: "compaction";
	summary: string;
	firstKeptEntryId: string;
	tokensBefore: number;
	/** 扩展专用数据（例如 ArtifactIndex、结构化压缩的版本标记） */
	details?: T;
	/** 生成该摘要的 LLM 调用所产生的用量信息（如有） */
	usage?: Usage;
	/** 由扩展生成时为 true；由 pi 生成时为 undefined/false（向后兼容） */
	fromHook?: boolean;
}

export interface BranchSummaryEntry<T = unknown> extends SessionEntryBase {
	type: "branch_summary";
	fromId: string;
	summary: string;
	/** 扩展专用数据（不发送给 LLM） */
	details?: T;
	/** 生成该摘要的 LLM 调用的用量信息（如有） */
	usage?: Usage;
	/** 由扩展生成时为 true，由 pi 生成时为 false */
	fromHook?: boolean;
}

/**
 * 供扩展在会话中存储扩展专用数据的自定义条目。
 * 使用 customType 标识你的扩展所写入的条目。
 *
 * 用途：在会话重新加载之间持久化扩展状态。重新加载时，扩展可以
 * 扫描属于自己的 customType 条目并重建内部状态。
 *
 * 不参与 LLM 上下文（会被 buildSessionContext 忽略）。
 * 若需向上下文注入内容，请参见 CustomMessageEntry。
 */
export interface CustomEntry<T = unknown> extends SessionEntryBase {
	type: "custom";
	customType: string;
	data?: T;
}

/** 标签条目，用于在条目上添加用户自定义的书签/标记。 */
export interface LabelEntry extends SessionEntryBase {
	type: "label";
	targetId: string;
	label: string | undefined;
}

/** 会话元数据条目（例如用户自定义的显示名称）。 */
export interface SessionInfoEntry extends SessionEntryBase {
	type: "session_info";
	name?: string;
}

/**
 * 供扩展向 LLM 上下文注入消息的自定义消息条目。
 * 使用 customType 标识你的扩展所写入的条目。
 *
 * 与 CustomEntry 不同，该条目会参与 LLM 上下文。
 * 其 content 会在 buildSessionContext() 中被转换为一条用户消息。
 * details 用于存放扩展专用元数据（不会发送给 LLM）。
 *
 * display 控制 TUI 渲染：
 * - false：完全隐藏
 * - true：使用专属样式渲染（与用户消息不同）
 */
export interface CustomMessageEntry<T = unknown> extends SessionEntryBase {
	type: "custom_message";
	customType: string;
	content: string | (TextContent | ImageContent)[];
	details?: T;
	display: boolean;
}

/** 会话条目——带有 id/parentId 以构成树结构（由 SessionManager 中各类“读取”方法返回） */
export type SessionEntry =
	| SessionMessageEntry
	| ThinkingLevelChangeEntry
	| ModelChangeEntry
	| CompactionEntry
	| BranchSummaryEntry
	| CustomEntry
	| CustomMessageEntry
	| LabelEntry
	| SessionInfoEntry;

/** 原始文件条目（包含头部） */
export type FileEntry = SessionHeader | SessionEntry;

/** getTree() 使用的树节点——会话结构的防御性拷贝 */
export interface SessionTreeNode {
	entry: SessionEntry;
	children: SessionTreeNode[];
	/** 该条目解析后的标签（如有） */
	label?: string;
	/** 该条目最近一次标签变更的时间戳（如有） */
	labelTimestamp?: string;
}

export interface SessionContext {
	messages: AgentMessage[];
	thinkingLevel: string;
	model: { provider: string; modelId: string } | null;
}

export interface SessionInfo {
	path: string;
	id: string;
	/** 会话启动时的工作目录。旧会话中该字段为空字符串。 */
	cwd: string;
	/** 来自 session_info 条目的用户自定义显示名称。 */
	name?: string;
	/** 父会话的路径（若该会话是通过 fork 创建的）。 */
	parentSessionPath?: string;
	created: Date;
	modified: Date;
	messageCount: number;
	firstMessage: string;
	allMessagesText: string;
}

export type ReadonlySessionManager = Pick<
	SessionManager,
	| "getCwd"
	| "getSessionDir"
	| "getSessionId"
	| "getSessionFile"
	| "getLeafId"
	| "getLeafEntry"
	| "getEntry"
	| "getLabel"
	| "getBranch"
	| "buildContextEntries"
	| "getHeader"
	| "getEntries"
	| "getTree"
	| "getSessionName"
>;

function createSessionId(): string {
	return uuidv7();
}

export function assertValidSessionId(id: string): void {
	if (!/^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/.test(id)) {
		throw new Error(
			"Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', and start and end with an alphanumeric character",
		);
	}
}

/** 生成唯一的短 ID（8 位十六进制字符，已做冲突检查） */
function generateId(byId: { has(id: string): boolean }): string {
	for (let i = 0; i < 100; i++) {
		const id = randomUUID().slice(0, 8);
		if (!byId.has(id)) return id;
	}
	// 万一发生冲突，回退到完整 UUID
	return randomUUID();
}

/** v1 → v2 迁移：补上 id/parentId 树结构。就地修改。 */
function migrateV1ToV2(entries: FileEntry[]): void {
	const ids = new Set<string>();
	let prevId: string | null = null;

	for (const entry of entries) {
		if (entry.type === "session") {
			entry.version = 2;
			continue;
		}

		entry.id = generateId(ids);
		entry.parentId = prevId;
		prevId = entry.id;

		// 将压缩条目的 firstKeptEntryIndex 转换为 firstKeptEntryId
		if (entry.type === "compaction") {
			const comp = entry as CompactionEntry & { firstKeptEntryIndex?: number };
			if (typeof comp.firstKeptEntryIndex === "number") {
				const targetEntry = entries[comp.firstKeptEntryIndex];
				if (targetEntry && targetEntry.type !== "session") {
					comp.firstKeptEntryId = targetEntry.id;
				}
				delete comp.firstKeptEntryIndex;
			}
		}
	}
}

/** v2 → v3 迁移：将 hookMessage 角色重命名为 custom。就地修改。 */
function migrateV2ToV3(entries: FileEntry[]): void {
	for (const entry of entries) {
		if (entry.type === "session") {
			entry.version = 3;
			continue;
		}

		// 更新角色为 hookMessage 的消息条目
		if (entry.type === "message") {
			const msgEntry = entry as SessionMessageEntry;
			if (msgEntry.message && (msgEntry.message as { role: string }).role === "hookMessage") {
				(msgEntry.message as { role: string }).role = "custom";
			}
		}
	}
}

/**
 * 执行必要的迁移，将条目升至当前版本。
 * 就地修改条目。若执行了任一迁移则返回 true。
 */
function migrateToCurrentVersion(entries: FileEntry[]): boolean {
	const header = entries.find((e) => e.type === "session") as SessionHeader | undefined;
	const version = header?.version ?? 1;

	if (version >= CURRENT_SESSION_VERSION) return false;

	if (version < 2) migrateV1ToV2(entries);
	if (version < 3) migrateV2ToV3(entries);

	return true;
}

/** 导出以便测试使用 */
export function migrateSessionEntries(entries: FileEntry[]): void {
	migrateToCurrentVersion(entries);
}

/** 导出以供 compaction.test.ts 使用 */
export function parseSessionEntries(content: string): FileEntry[] {
	const entries: FileEntry[] = [];
	const lines = content.trim().split("\n");

	for (const line of lines) {
		if (!line.trim()) continue;
		try {
			const entry = JSON.parse(line) as FileEntry;
			entries.push(entry);
		} catch {
			// 跳过格式错误的行
		}
	}

	return entries;
}

export function getLatestCompactionEntry(entries: SessionEntry[]): CompactionEntry | null {
	for (let i = entries.length - 1; i >= 0; i--) {
		if (entries[i].type === "compaction") {
			return entries[i] as CompactionEntry;
		}
	}
	return null;
}

function buildEntryIndex(entries: SessionEntry[], byId?: Map<string, SessionEntry>): Map<string, SessionEntry> {
	if (byId) return byId;
	const index = new Map<string, SessionEntry>();
	for (const entry of entries) {
		index.set(entry.id, entry);
	}
	return index;
}

function buildSessionPath(
	entries: SessionEntry[],
	leafId?: string | null,
	byId?: Map<string, SessionEntry>,
): SessionEntry[] {
	const index = buildEntryIndex(entries, byId);
	let leaf: SessionEntry | undefined;
	if (leafId === null) {
		return [];
	}
	if (leafId) {
		leaf = index.get(leafId);
	}
	leaf ??= entries[entries.length - 1];
	if (!leaf) {
		return [];
	}

	const path: SessionEntry[] = [];
	let current: SessionEntry | undefined = leaf;
	while (current) {
		path.push(current);
		current = current.parentId ? index.get(current.parentId) : undefined;
	}
	path.reverse();
	return path;
}

function getSessionContextSettings(path: SessionEntry[]): Pick<SessionContext, "thinkingLevel" | "model"> {
	let thinkingLevel = "off";
	let model: { provider: string; modelId: string } | null = null;

	for (const entry of path) {
		if (entry.type === "thinking_level_change") {
			thinkingLevel = entry.thinkingLevel;
		} else if (entry.type === "model_change") {
			model = { provider: entry.provider, modelId: entry.modelId };
		} else if (entry.type === "message" && entry.message.role === "assistant") {
			model = { provider: entry.message.provider, modelId: entry.message.model };
		}
	}

	return { thinkingLevel, model };
}

/**
 * 将一个选定的会话条目投影为 LLM/运行时消息。
 * 普通的 custom 条目属于展示/状态条目，不参与上下文。
 */
export function sessionEntryToContextMessages(entry: SessionEntry): AgentMessage[] {
	if (entry.type === "message") {
		const message = entry.message;
		// 会话文件解析时不做校验；旧版本、fork 出来的或
		// 手工编辑过的文件可能包含 content 为 null 或缺失的消息。
		if (
			(message.role === "user" || message.role === "assistant" || message.role === "toolResult") &&
			message.content == null
		) {
			return [{ ...message, content: [] }];
		}
		return [message];
	}
	if (entry.type === "custom_message") {
		return [
			createCustomMessage(entry.customType, entry.content ?? [], entry.display, entry.details, entry.timestamp),
		];
	}
	if (entry.type === "branch_summary" && entry.summary) {
		return [createBranchSummaryMessage(entry.summary, entry.fromId, entry.timestamp)];
	}
	if (entry.type === "compaction") {
		return [createCompactionSummaryMessage(entry.summary, entry.tokensBefore, entry.timestamp)];
	}
	return [];
}

/**
 * 构建当前生效且感知压缩的会话条目列表。
 *
 * 该列表沿当前叶子路径展开。若路径中存在压缩条目，
 * 则最近的压缩由压缩条目本身表示，其后依次是
 * 从 firstKeptEntryId 起保留的条目，以及压缩条目之后的所有条目。
 * 更早的、已被摘要的条目会被省略。
 */
export function buildContextEntries(
	entries: SessionEntry[],
	leafId?: string | null,
	byId?: Map<string, SessionEntry>,
): SessionEntry[] {
	const path = buildSessionPath(entries, leafId, byId);
	let compaction: CompactionEntry | null = null;

	for (const entry of path) {
		if (entry.type === "compaction") {
			compaction = entry;
		}
	}

	if (!compaction) {
		return path;
	}

	const compactionIdx = path.findIndex((entry) => entry.id === compaction.id);
	if (compactionIdx < 0) {
		return path;
	}

	const contextEntries: SessionEntry[] = [compaction];
	let foundFirstKept = false;
	for (let i = 0; i < compactionIdx; i++) {
		const entry = path[i];
		if (entry.id === compaction.firstKeptEntryId) {
			foundFirstKept = true;
		}
		if (foundFirstKept) {
			contextEntries.push(entry);
		}
	}
	contextEntries.push(...path.slice(compactionIdx + 1));
	return contextEntries;
}

/**
 * 通过树遍历从条目构建会话上下文。
 * 若提供了 leafId，则从该条目向上走到根。
 * 会处理路径上的压缩与分支摘要。
 */
export function buildSessionContext(
	entries: SessionEntry[],
	leafId?: string | null,
	byId?: Map<string, SessionEntry>,
): SessionContext {
	const path = buildSessionPath(entries, leafId, byId);
	const { thinkingLevel, model } = getSessionContextSettings(path);
	const messages = buildContextEntries(entries, leafId, byId).flatMap(sessionEntryToContextMessages);
	return { messages, thinkingLevel, model };
}

/**
 * 计算某个 cwd 对应的默认会话目录。
 * 将 cwd 编码为 ~/.pi/agent/sessions/ 下的安全目录名。
 */
function getDefaultSessionDirPath(cwd: string, agentDir: string = getDefaultAgentDir()): string {
	const resolvedCwd = resolvePath(cwd);
	const resolvedAgentDir = resolvePath(agentDir);
	const safePath = `--${resolvedCwd.replace(/^[/\\]/, "").replace(/[/\\:]/g, "-")}--`;
	return join(resolvedAgentDir, "sessions", safePath);
}

export function getDefaultSessionDir(cwd: string, agentDir: string = getDefaultAgentDir()): string {
	const sessionDir = getDefaultSessionDirPath(cwd, agentDir);
	if (!existsSync(sessionDir)) {
		mkdirSync(sessionDir, { recursive: true });
	}
	return sessionDir;
}

const SESSION_READ_BUFFER_SIZE = 1024 * 1024;
const SESSION_HEADER_READ_BUFFER_SIZE = 4096;
/** 限制同步头部发现过程的读取量，同时兼顾较大的 cwd 与自定义元数据字段。 */
const MAX_SESSION_HEADER_SCAN_BYTES = 1024 * 1024;

class SessionHeaderScanLimitError extends Error {
	constructor(filePath: string) {
		super(`Session header exceeds ${MAX_SESSION_HEADER_SCAN_BYTES}-byte scan limit: ${filePath}`);
		this.name = "SessionHeaderScanLimitError";
	}
}

function parseSessionEntryLine(line: string): FileEntry | null {
	if (!line.trim()) return null;
	try {
		return JSON.parse(line) as FileEntry;
	} catch {
		// 跳过格式错误的行
		return null;
	}
}

/** 导出以便测试使用 */
export function loadEntriesFromFile(filePath: string): FileEntry[] {
	const resolvedFilePath = normalizePath(filePath);
	if (!existsSync(resolvedFilePath)) return [];

	const entries: FileEntry[] = [];
	let pending = "";
	const fd = openSync(resolvedFilePath, "r");
	try {
		const decoder = new StringDecoder("utf8");
		const buffer = Buffer.allocUnsafe(SESSION_READ_BUFFER_SIZE);

		while (true) {
			const bytesRead = readSync(fd, buffer, 0, buffer.length, null);
			if (bytesRead === 0) break;

			pending += decoder.write(buffer.subarray(0, bytesRead));
			let lineStart = 0;
			let newlineIndex = pending.indexOf("\n", lineStart);
			while (newlineIndex !== -1) {
				const entry = parseSessionEntryLine(pending.slice(lineStart, newlineIndex));
				if (entry) entries.push(entry);
				lineStart = newlineIndex + 1;
				newlineIndex = pending.indexOf("\n", lineStart);
			}
			pending = pending.slice(lineStart);
		}

		pending += decoder.end();
		const finalEntry = parseSessionEntryLine(pending);
		if (finalEntry) entries.push(finalEntry);
	} finally {
		closeSync(fd);
	}

	// 在修复文件之前先校验会话头部。
	if (entries.length === 0) return entries;
	const header = entries[0];
	if (header.type !== "session" || typeof (header as { id?: unknown }).id !== "string") {
		return [];
	}

	if (pending) appendFileSync(resolvedFilePath, "\n");
	return entries;
}

/**
 * 在查找首个可解析的会话条目时检查单个物理行。
 * 空行与格式错误的行会被跳过，与 loadEntriesFromFile() 的行为保持一致。
 * 返回 undefined 表示继续扫描，返回 null 表示解析出了非头部条目，否则返回头部。
 */
function parseSessionHeaderCandidate(line: string): SessionHeader | null | undefined {
	if (!line.trim()) return undefined;
	const entry = parseSessionEntryLine(line);
	if (!entry) return undefined;
	if (entry.type !== "session" || typeof (entry as { id?: unknown }).id !== "string") return null;
	return entry;
}

function readSessionHeader(filePath: string): SessionHeader | null {
	const fd = openSync(filePath, "r");
	try {
		const decoder = new StringDecoder("utf8");
		const buffer = Buffer.allocUnsafe(SESSION_HEADER_READ_BUFFER_SIZE);
		const lineChunks: string[] = [];
		let scannedBytes = 0;

		while (scannedBytes < MAX_SESSION_HEADER_SCAN_BYTES) {
			const readLength = Math.min(buffer.length, MAX_SESSION_HEADER_SCAN_BYTES - scannedBytes);
			const bytesRead = readSync(fd, buffer, 0, readLength, null);
			if (bytesRead === 0) {
				lineChunks.push(decoder.end());
				return parseSessionHeaderCandidate(lineChunks.join("")) ?? null;
			}
			scannedBytes += bytesRead;

			const chunk = decoder.write(buffer.subarray(0, bytesRead));
			let lineStart = 0;
			let newlineIndex = chunk.indexOf("\n", lineStart);
			while (newlineIndex !== -1) {
				lineChunks.push(chunk.slice(lineStart, newlineIndex));
				const header = parseSessionHeaderCandidate(lineChunks.join(""));
				if (header !== undefined) return header;
				lineChunks.length = 0;
				lineStart = newlineIndex + 1;
				newlineIndex = chunk.indexOf("\n", lineStart);
			}
			lineChunks.push(chunk.slice(lineStart));
		}

		// 探测是否已到 EOF，这样当末尾头部不带换行且恰好结束在
		// 扫描上限处时也能被接受。只要再多一个字节就超出受限扫描范围。
		const probe = Buffer.allocUnsafe(1);
		if (readSync(fd, probe, 0, probe.length, null) === 0) {
			lineChunks.push(decoder.end());
			return parseSessionHeaderCandidate(lineChunks.join("")) ?? null;
		}
		throw new SessionHeaderScanLimitError(filePath);
	} finally {
		closeSync(fd);
	}
}

function readSessionHeaderForDiscovery(filePath: string): SessionHeader | null {
	try {
		return readSessionHeader(filePath);
	} catch {
		// 发现过程是尽力而为的：不可读或过大的文件不算会话，
		// 且单个损坏文件不应妨碍其他会话被发现。
		return null;
	}
}

function getSessionHeaderCwd(header: SessionHeader): string | undefined {
	const cwd = (header as { cwd?: unknown }).cwd;
	return typeof cwd === "string" ? cwd : undefined;
}

function sessionCwdMatches(cwd: string | undefined, resolvedCwd: string): boolean {
	return cwd !== undefined && cwd !== "" && resolvePath(cwd) === resolvedCwd;
}

/** 导出以便测试使用 */
export function findMostRecentSession(sessionDir: string, cwd?: string): string | null {
	const resolvedSessionDir = normalizePath(sessionDir);
	const resolvedCwd = cwd ? resolvePath(cwd) : undefined;
	try {
		const files = readdirSync(resolvedSessionDir)
			.filter((f) => f.endsWith(".jsonl"))
			.map((f) => join(resolvedSessionDir, f))
			.map((path) => ({ path, header: readSessionHeaderForDiscovery(path) }))
			.filter(
				(file): file is { path: string; header: SessionHeader } =>
					file.header !== null &&
					(!resolvedCwd || sessionCwdMatches(getSessionHeaderCwd(file.header), resolvedCwd)),
			)
			.map(({ path }) => ({ path, mtime: statSync(path).mtime }))
			.sort((a, b) => b.mtime.getTime() - a.mtime.getTime());

		return files[0]?.path || null;
	} catch {
		// 目录访问与 stat 竞争会导致无法发现最近会话。
		return null;
	}
}

function isMessageWithContent(message: AgentMessage): message is Message {
	return typeof (message as Message).role === "string" && "content" in message;
}

function extractTextContent(message: Message): string {
	const content = message.content;
	if (typeof content === "string") {
		return content;
	}
	return content
		.filter((block): block is TextContent => block.type === "text")
		.map((block) => block.text)
		.join(" ");
}

function getMessageActivityTime(entry: SessionMessageEntry): number | undefined {
	const message = entry.message;
	if (!isMessageWithContent(message)) return undefined;
	if (message.role !== "user" && message.role !== "assistant") return undefined;

	const msgTimestamp = (message as { timestamp?: number }).timestamp;
	if (typeof msgTimestamp === "number") {
		return msgTimestamp;
	}

	const t = new Date(entry.timestamp).getTime();
	return Number.isNaN(t) ? undefined : t;
}

async function buildSessionInfo(filePath: string): Promise<SessionInfo | null> {
	try {
		const stats = await stat(filePath);
		let header: SessionHeader | null = null;
		let messageCount = 0;
		let firstMessage = "";
		const allMessages: string[] = [];
		let name: string | undefined;
		let lastActivityTime: number | undefined;

		const rl = createInterface({
			input: createReadStream(filePath, { encoding: "utf8" }),
			crlfDelay: Infinity,
		});

		for await (const line of rl) {
			const entry = parseSessionEntryLine(line);
			if (!entry) continue;

			if (!header) {
				if (entry.type !== "session") return null;
				header = entry;
				continue;
			}

			// 提取会话名称（使用最新的，包括显式的清空操作）
			if (entry.type === "session_info") {
				name = entry.name?.trim() || undefined;
			}

			if (entry.type !== "message") continue;
			messageCount++;

			const activityTime = getMessageActivityTime(entry);
			if (typeof activityTime === "number") {
				lastActivityTime = Math.max(lastActivityTime ?? 0, activityTime);
			}

			const message = entry.message;
			if (!isMessageWithContent(message)) continue;
			if (message.role !== "user" && message.role !== "assistant") continue;

			const textContent = extractTextContent(message);
			if (!textContent) continue;

			allMessages.push(textContent);
			if (!firstMessage && message.role === "user") {
				firstMessage = textContent;
			}
		}

		if (!header) return null;

		const cwd = typeof header.cwd === "string" ? header.cwd : "";
		const parentSessionPath = header.parentSession;
		const headerTime = typeof header.timestamp === "string" ? new Date(header.timestamp).getTime() : NaN;
		const modified =
			typeof lastActivityTime === "number" && lastActivityTime > 0
				? new Date(lastActivityTime)
				: !Number.isNaN(headerTime)
					? new Date(headerTime)
					: stats.mtime;

		return {
			path: filePath,
			id: header.id,
			cwd,
			name,
			parentSessionPath,
			created: new Date(header.timestamp),
			modified,
			messageCount,
			firstMessage: firstMessage || "(no messages)",
			allMessagesText: allMessages.join(" "),
		};
	} catch {
		return null;
	}
}

export type SessionListProgress = (loaded: number, total: number) => void;

const MAX_CONCURRENT_SESSION_INFO_LOADS = 10;

async function buildSessionInfosWithConcurrency(
	files: string[],
	onLoaded: () => void,
): Promise<(SessionInfo | null)[]> {
	const results: (SessionInfo | null)[] = new Array(files.length).fill(null);
	const inFlight = new Set<Promise<void>>();
	let nextIndex = 0;

	const startNext = (): void => {
		const index = nextIndex++;
		const file = files[index];
		if (!file) return;

		let task: Promise<void>;
		task = buildSessionInfo(file)
			.then((info) => {
				results[index] = info;
			})
			.catch(() => {
				results[index] = null;
			})
			.finally(() => {
				inFlight.delete(task);
				onLoaded();
			});
		inFlight.add(task);
	};

	while (nextIndex < files.length || inFlight.size > 0) {
		while (nextIndex < files.length && inFlight.size < MAX_CONCURRENT_SESSION_INFO_LOADS) {
			startNext();
		}
		if (inFlight.size > 0) {
			await Promise.race(inFlight);
		}
	}

	return results;
}

async function listSessionsFromDir(
	dir: string,
	onProgress?: SessionListProgress,
	progressOffset = 0,
	progressTotal?: number,
): Promise<SessionInfo[]> {
	const sessions: SessionInfo[] = [];
	if (!existsSync(dir)) {
		return sessions;
	}

	try {
		const dirEntries = await readdir(dir);
		const files = dirEntries.filter((f) => f.endsWith(".jsonl")).map((f) => join(dir, f));
		const total = progressTotal ?? files.length;

		let loaded = 0;
		const results = await buildSessionInfosWithConcurrency(files, () => {
			loaded++;
			onProgress?.(progressOffset + loaded, total);
		});
		for (const info of results) {
			if (info) {
				sessions.push(info);
			}
		}
	} catch {
		// 出错时返回空列表
	}

	return sessions;
}

/**
 * 以仅追加的树结构管理对话会话，并将其存储在 JSONL 文件中。
 *
 * 每个会话条目都有 id 和 parentId，构成树结构。“leaf”（叶子）指针
 * 记录当前位置。追加条目会在当前叶子下创建子条目。
 * 分支会将叶子移动到更早的条目，从而在不修改历史的前提下创建新分支。
 *
 * 使用 buildSessionContext() 获取解析后供 LLM 使用的消息列表，
 * 它会处理压缩摘要，并沿从根到当前叶子的路径展开。
 */
export class SessionManager {
	private sessionId: string = "";
	private sessionFile: string | undefined;
	private sessionDir: string;
	private cwd: string;
	private persist: boolean;
	private flushed: boolean = false;
	private fileEntries: FileEntry[] = [];
	private byId: Map<string, SessionEntry> = new Map();
	private labelsById: Map<string, string> = new Map();
	private labelTimestampsById: Map<string, string> = new Map();
	private leafId: string | null = null;

	private constructor(
		cwd: string,
		sessionDir: string,
		sessionFile: string | undefined,
		persist: boolean,
		newSessionOptions?: NewSessionOptions,
		preloadedFileEntries?: FileEntry[],
	) {
		this.cwd = resolvePath(cwd);
		this.sessionDir = normalizePath(sessionDir);
		this.persist = persist;
		if (persist && this.sessionDir && !existsSync(this.sessionDir)) {
			mkdirSync(this.sessionDir, { recursive: true });
		}

		if (sessionFile) {
			this._setSessionFile(sessionFile, preloadedFileEntries);
		} else if (preloadedFileEntries?.length) {
			this._loadEntries(preloadedFileEntries, newSessionOptions);
		} else {
			this.newSession(newSessionOptions);
		}
	}

	/** 切换到另一个会话文件（用于恢复与分支） */
	setSessionFile(sessionFile: string): void {
		this._setSessionFile(sessionFile);
	}

	private _setSessionFile(sessionFile: string, preloadedFileEntries?: FileEntry[]): void {
		this.sessionFile = resolvePath(sessionFile);
		if (existsSync(this.sessionFile)) {
			const entries = preloadedFileEntries ?? loadEntriesFromFile(this.sessionFile);

			// 若文件为空，则用有效的会话头部初始化它。若非空但无法
			// 解析为 pi 会话，则报错且不修改该文件。
			if (entries.length === 0) {
				const explicitPath = this.sessionFile;
				if (statSync(explicitPath).size > 0) {
					throw new Error(`Session file is not a valid ${APP_NAME} session: ${explicitPath}`);
				}
				this.newSession();
				this.sessionFile = explicitPath;
				this._rewriteFile();
				this.flushed = true;
				return;
			}

			this._loadEntries(entries);
			this.flushed = true;
		} else {
			const explicitPath = this.sessionFile;
			this.newSession();
			this.sessionFile = explicitPath; // 保留来自 --session 标志的显式路径
		}
	}

	newSession(options?: NewSessionOptions): string | undefined {
		if (options?.id !== undefined) {
			assertValidSessionId(options.id);
		}
		this.sessionId = options?.id ?? createSessionId();
		const timestamp = new Date().toISOString();
		const header: SessionHeader = {
			type: "session",
			version: CURRENT_SESSION_VERSION,
			id: this.sessionId,
			timestamp,
			cwd: this.cwd,
			parentSession: options?.parentSession,
		};
		this.fileEntries = [header];
		this.byId.clear();
		this.labelsById.clear();
		this.labelTimestampsById.clear();
		this.leafId = null;
		this.flushed = false;

		if (this.persist) {
			const fileTimestamp = timestamp.replace(/[:.]/g, "-");
			this.sessionFile = join(this.getSessionDir(), `${fileTimestamp}_${this.sessionId}.jsonl`);
		}
		return this.sessionFile;
	}

	private _loadEntries(entries: FileEntry[], options?: NewSessionOptions): void {
		const header = entries.find((e) => e.type === "session") as SessionHeader | undefined;

		if (header) {
			this.fileEntries = entries;
			this.sessionId = header.id;

			if (migrateToCurrentVersion(this.fileEntries)) {
				this._rewriteFile();
			}
		} else {
			this.newSession(options);
			this.fileEntries = this.fileEntries.concat(entries);
		}

		this._buildIndex();
	}

	private _buildIndex(): void {
		this.byId.clear();
		this.labelsById.clear();
		this.labelTimestampsById.clear();
		this.leafId = null;
		for (const entry of this.fileEntries) {
			if (entry.type === "session") continue;
			this.byId.set(entry.id, entry);
			this.leafId = entry.id;
			if (entry.type === "label") {
				if (entry.label) {
					this.labelsById.set(entry.targetId, entry.label);
					this.labelTimestampsById.set(entry.targetId, entry.timestamp);
				} else {
					this.labelsById.delete(entry.targetId);
					this.labelTimestampsById.delete(entry.targetId);
				}
			}
		}
	}

	private _rewriteFile(): void {
		if (!this.persist || !this.sessionFile) return;
		const fd = openSync(this.sessionFile, "w");
		try {
			for (const entry of this.fileEntries) {
				writeFileSync(fd, `${JSON.stringify(entry)}\n`);
			}
		} finally {
			closeSync(fd);
		}
	}

	isPersisted(): boolean {
		return this.persist;
	}

	getCwd(): string {
		return this.cwd;
	}

	getSessionDir(): string {
		return this.sessionDir;
	}

	usesDefaultSessionDir(): boolean {
		return this.sessionDir === getDefaultSessionDirPath(this.cwd);
	}

	getSessionId(): string {
		return this.sessionId;
	}

	getSessionFile(): string | undefined {
		return this.sessionFile;
	}

	_persist(entry: SessionEntry): void {
		if (!this.persist || !this.sessionFile) return;

		const hasAssistant = this.fileEntries.some((e) => e.type === "message" && e.message.role === "assistant");
		if (!hasAssistant) {
			if (this.flushed) {
				appendFileSync(this.sessionFile, `${JSON.stringify(entry)}\n`);
			} else {
				// 标记为未刷新，这样当 assistant 消息到来时，所有条目都会被写入
				this.flushed = false;
			}
			return;
		}

		if (!this.flushed) {
			const fd = openSync(this.sessionFile, "wx");
			try {
				for (const e of this.fileEntries) {
					writeFileSync(fd, `${JSON.stringify(e)}\n`);
				}
			} finally {
				closeSync(fd);
			}
			this.flushed = true;
		} else {
			appendFileSync(this.sessionFile, `${JSON.stringify(entry)}\n`);
		}
	}

	private _appendEntry(entry: SessionEntry): void {
		this.fileEntries.push(entry);
		this.byId.set(entry.id, entry);
		this.leafId = entry.id;
		this._persist(entry);
	}

	/** 将一条消息作为当前叶子的子条目追加，然后推进叶子。返回条目 id。
	 * 不允许直接写入 CompactionSummaryMessage 和 BranchSummaryMessage。
	 * 原因：我们希望它们是会话中的顶层条目，而不是消息类型的会话条目，
	 * 这样更容易找到它们。
	 * 这些内容需要通过 appendCompaction() 和 appendBranchSummary() 方法追加。
	 */
	appendMessage(message: Message | CustomMessage | BashExecutionMessage): string {
		const entry: SessionMessageEntry = {
			type: "message",
			id: generateId(this.byId),
			parentId: this.leafId,
			timestamp: new Date().toISOString(),
			message,
		};
		this._appendEntry(entry);
		return entry.id;
	}

	/** 将一次思考等级变更作为当前叶子的子条目追加，然后推进叶子。返回条目 id。 */
	appendThinkingLevelChange(thinkingLevel: string): string {
		const entry: ThinkingLevelChangeEntry = {
			type: "thinking_level_change",
			id: generateId(this.byId),
			parentId: this.leafId,
			timestamp: new Date().toISOString(),
			thinkingLevel,
		};
		this._appendEntry(entry);
		return entry.id;
	}

	/** 将一次模型变更作为当前叶子的子条目追加，然后推进叶子。返回条目 id。 */
	appendModelChange(provider: string, modelId: string): string {
		const entry: ModelChangeEntry = {
			type: "model_change",
			id: generateId(this.byId),
			parentId: this.leafId,
			timestamp: new Date().toISOString(),
			provider,
			modelId,
		};
		this._appendEntry(entry);
		return entry.id;
	}

	/** 将一条压缩摘要作为当前叶子的子条目追加，然后推进叶子。返回条目 id。 */
	appendCompaction<T = unknown>(
		summary: string,
		firstKeptEntryId: string,
		tokensBefore: number,
		details?: T,
		fromHook?: boolean,
		usage?: Usage,
	): string {
		const entry: CompactionEntry<T> = {
			type: "compaction",
			id: generateId(this.byId),
			parentId: this.leafId,
			timestamp: new Date().toISOString(),
			summary,
			firstKeptEntryId,
			tokensBefore,
			details,
			usage,
			fromHook,
		};
		this._appendEntry(entry);
		return entry.id;
	}

	/** 将一个自定义条目（供扩展使用）作为当前叶子的子条目追加，然后推进叶子。返回条目 id。 */
	appendCustomEntry(customType: string, data?: unknown): string {
		const entry: CustomEntry = {
			type: "custom",
			customType,
			data,
			id: generateId(this.byId),
			parentId: this.leafId,
			timestamp: new Date().toISOString(),
		};
		this._appendEntry(entry);
		return entry.id;
	}

	/** 追加一条会话信息条目（例如显示名称）。返回条目 id。 */
	appendSessionInfo(name: string): string {
		const sanitizedName = name.replace(/[\r\n]+/g, " ").trim();
		const entry: SessionInfoEntry = {
			type: "session_info",
			id: generateId(this.byId),
			parentId: this.leafId,
			timestamp: new Date().toISOString(),
			name: sanitizedName,
		};
		this._appendEntry(entry);
		return entry.id;
	}

	/** 从最新的 session_info 条目获取当前会话名称（如有）。 */
	getSessionName(): string | undefined {
		// 倒序遍历条目以找到最新的 session_info 条目。
		// 空名称会显式清除会话标题。
		const entries = this.getEntries();
		for (let i = entries.length - 1; i >= 0; i--) {
			const entry = entries[i];
			if (entry.type === "session_info") {
				return entry.name?.trim() || undefined;
			}
		}
		return undefined;
	}

	/**
	 * 追加一个参与 LLM 上下文的自定义消息条目（供扩展使用）。
	 * @param customType 扩展标识符，用于在重新加载时筛选
	 * @param content 消息内容（字符串或 TextContent/ImageContent 数组）
	 * @param display 是否在 TUI 中显示（true = 以专属样式显示，false = 隐藏）
	 * @param details 可选的扩展专用元数据（不发送给 LLM）
	 * @returns 条目 id
	 */
	appendCustomMessageEntry<T = unknown>(
		customType: string,
		content: string | (TextContent | ImageContent)[],
		display: boolean,
		details?: T,
	): string {
		const entry: CustomMessageEntry<T> = {
			type: "custom_message",
			customType,
			content,
			display,
			details,
			id: generateId(this.byId),
			parentId: this.leafId,
			timestamp: new Date().toISOString(),
		};
		this._appendEntry(entry);
		return entry.id;
	}

	// =========================================================================
	// 树遍历
	// =========================================================================

	getLeafId(): string | null {
		return this.leafId;
	}

	getLeafEntry(): SessionEntry | undefined {
		return this.leafId ? this.byId.get(this.leafId) : undefined;
	}

	getEntry(id: string): SessionEntry | undefined {
		return this.byId.get(id);
	}

	/**
	 * 获取某个条目的所有直接子条目。
	 */
	getChildren(parentId: string): SessionEntry[] {
		const children: SessionEntry[] = [];
		for (const entry of this.byId.values()) {
			if (entry.parentId === parentId) {
				children.push(entry);
			}
		}
		return children;
	}

	/**
	 * 获取条目的标签（如有）。
	 */
	getLabel(id: string): string | undefined {
		return this.labelsById.get(id);
	}

	/**
	 * 设置或清除条目的标签。
	 * 标签是用户自定义的标记，用于书签/导航。
	 * 传入 undefined 或空字符串即可清除标签。
	 */
	appendLabelChange(targetId: string, label: string | undefined): string {
		if (!this.byId.has(targetId)) {
			throw new Error(`Entry ${targetId} not found`);
		}
		const entry: LabelEntry = {
			type: "label",
			id: generateId(this.byId),
			parentId: this.leafId,
			timestamp: new Date().toISOString(),
			targetId,
			label,
		};
		this._appendEntry(entry);
		if (label) {
			this.labelsById.set(targetId, label);
			this.labelTimestampsById.set(targetId, entry.timestamp);
		} else {
			this.labelsById.delete(targetId);
			this.labelTimestampsById.delete(targetId);
		}
		return entry.id;
	}

	/**
	 * 从条目向上走到根，按路径顺序返回所有条目。
	 * 包含所有条目类型（消息、压缩、模型变更等）。
	 * 使用 buildSessionContext() 获取解析后供 LLM 使用的消息。
	 */
	getBranch(fromId?: string): SessionEntry[] {
		const path: SessionEntry[] = [];
		const startId = fromId ?? this.leafId;
		let current = startId ? this.byId.get(startId) : undefined;
		while (current) {
			path.push(current);
			current = current.parentId ? this.byId.get(current.parentId) : undefined;
		}
		path.reverse();
		return path;
	}

	/**
	 * 构建当前生效且感知压缩的条目列表，用于上下文/渲染。
	 * 从当前叶子开始进行树遍历。
	 */
	buildContextEntries(): SessionEntry[] {
		return buildContextEntries(this.getEntries(), this.leafId, this.byId);
	}

	/**
	 * 构建会话上下文（即发送给 LLM 的内容）。
	 * 从当前叶子开始进行树遍历。
	 */
	buildSessionContext(): SessionContext {
		return buildSessionContext(this.getEntries(), this.leafId, this.byId);
	}

	/**
	 * 获取会话头部。
	 */
	getHeader(): SessionHeader | null {
		const h = this.fileEntries.find((e) => e.type === "session");
		return h ? (h as SessionHeader) : null;
	}

	/**
	 * 获取所有会话条目（不含头部）。返回浅拷贝。
	 * 会话是仅追加的：用 appendXXX() 添加条目，用 branch() 改变叶子指针。
	 * 条目不能被修改或删除。
	 */
	getEntries(): SessionEntry[] {
		return this.fileEntries.filter((e): e is SessionEntry => e.type !== "session");
	}

	/**
	 * 以树结构获取会话。返回所有条目的浅层防御性拷贝。
	 * 结构良好的会话恰好有一个根（首个 parentId === null 的条目）。
	 * 孤立条目（父链断裂）也会作为根返回。
	 */
	getTree(): SessionTreeNode[] {
		const entries = this.getEntries();
		const nodeMap = new Map<string, SessionTreeNode>();
		const roots: SessionTreeNode[] = [];

		// 创建带已解析标签的节点
		for (const entry of entries) {
			const label = this.labelsById.get(entry.id);
			const labelTimestamp = this.labelTimestampsById.get(entry.id);
			nodeMap.set(entry.id, { entry, children: [], label, labelTimestamp });
		}

		// 构建树
		for (const entry of entries) {
			const node = nodeMap.get(entry.id)!;
			if (entry.parentId === null || entry.parentId === entry.id) {
				roots.push(node);
			} else {
				const parent = nodeMap.get(entry.parentId);
				if (parent) {
					parent.children.push(node);
				} else {
					// 孤立条目——视为根
					roots.push(node);
				}
			}
		}

		// 按时间戳对子节点排序（最早在前，最新在底部）
		// 使用迭代方式以避免深层树导致的栈溢出
		const stack: SessionTreeNode[] = [...roots];
		while (stack.length > 0) {
			const node = stack.pop()!;
			node.children.sort((a, b) => new Date(a.entry.timestamp).getTime() - new Date(b.entry.timestamp).getTime());
			stack.push(...node.children);
		}

		return roots;
	}

	// =========================================================================
	// 分支
	// =========================================================================

	/**
	 * 从更早的条目开始一个新分支。
	 * 将叶子指针移动到指定条目。下一次调用 appendXXX() 时
	 * 会在该条目下创建子条目，从而形成新分支。已有条目
	 * 不会被修改或删除。
	 */
	branch(branchFromId: string): void {
		if (!this.byId.has(branchFromId)) {
			throw new Error(`Entry ${branchFromId} not found`);
		}
		this.leafId = branchFromId;
	}

	/**
	 * 将叶子指针重置为 null（即所有条目之前）。
	 * 下一次调用 appendXXX() 时会创建一个新的根条目（parentId = null）。
	 * 当导航以重新编辑第一条用户消息时使用。
	 */
	resetLeaf(): void {
		this.leafId = null;
	}

	/**
	 * 从被放弃的路径开始一个新分支，并附带其摘要。
	 * 与 branch() 相同，但会额外追加一条 branch_summary 条目，
	 * 用于记录被放弃的对话路径中的上下文。
	 */
	branchWithSummary(
		branchFromId: string | null,
		summary: string,
		details?: unknown,
		fromHook?: boolean,
		usage?: Usage,
	): string {
		if (branchFromId !== null && !this.byId.has(branchFromId)) {
			throw new Error(`Entry ${branchFromId} not found`);
		}
		const fromId = this.leafId ?? "root";
		this.leafId = branchFromId;
		const entry: BranchSummaryEntry = {
			type: "branch_summary",
			id: generateId(this.byId),
			parentId: branchFromId,
			timestamp: new Date().toISOString(),
			fromId,
			summary,
			details,
			usage,
			fromHook,
		};
		this._appendEntry(entry);
		return entry.id;
	}

	/**
	 * 创建一个新的会话文件，仅包含从根到指定叶子的路径。
	 * 适用于从带分支的会话中提取单条对话路径。
	 * 返回新会话文件路径；若不落盘则返回 undefined。
	 */
	createBranchedSession(leafId: string): string | undefined {
		const previousSessionFile = this.sessionFile;
		const path = this.getBranch(leafId);
		if (path.length === 0) {
			throw new Error(`Entry ${leafId} not found`);
		}

		// 从路径中过滤掉 LabelEntry——稍后会依据已解析的映射重新创建它们。
		// 由于标签是真实的树条目，后续条目可能是标签的子条目；
		// 移除标签需要重新串联保留下来的路径，以避免出现孤立的子树。
		const pathWithoutLabels: SessionEntry[] = [];
		const replacementByLabelId = new Map<string, string>();
		const pendingLabelIds: string[] = [];
		let pathParentId: string | null = null;
		for (const entry of path) {
			if (entry.type === "label") {
				pendingLabelIds.push(entry.id);
				continue;
			}
			for (const labelId of pendingLabelIds) {
				replacementByLabelId.set(labelId, entry.id);
			}
			pendingLabelIds.length = 0;
			pathWithoutLabels.push(
				entry.type === "compaction"
					? {
							...entry,
							parentId: pathParentId,
							firstKeptEntryId: replacementByLabelId.get(entry.firstKeptEntryId) ?? entry.firstKeptEntryId,
						}
					: { ...entry, parentId: pathParentId },
			);
			pathParentId = entry.id;
		}

		const newSessionId = createSessionId();
		const timestamp = new Date().toISOString();
		const fileTimestamp = timestamp.replace(/[:.]/g, "-");
		const newSessionFile = join(this.getSessionDir(), `${fileTimestamp}_${newSessionId}.jsonl`);

		const header: SessionHeader = {
			type: "session",
			version: CURRENT_SESSION_VERSION,
			id: newSessionId,
			timestamp,
			cwd: this.cwd,
			parentSession: this.persist ? previousSessionFile : undefined,
		};

		// 收集路径中条目对应的标签
		const pathEntryIds = new Set(pathWithoutLabels.map((e) => e.id));
		const labelsToWrite: Array<{ targetId: string; label: string; timestamp: string }> = [];
		for (const [targetId, label] of this.labelsById) {
			if (pathEntryIds.has(targetId)) {
				labelsToWrite.push({ targetId, label, timestamp: this.labelTimestampsById.get(targetId)! });
			}
		}

		if (this.persist) {
			// 构建标签条目
			const lastEntryId = pathWithoutLabels[pathWithoutLabels.length - 1]?.id || null;
			let parentId = lastEntryId;
			const labelEntries: LabelEntry[] = [];
			for (const { targetId, label, timestamp: labelTimestamp } of labelsToWrite) {
				const labelEntry: LabelEntry = {
					type: "label",
					id: generateId(new Set(pathEntryIds)),
					parentId,
					timestamp: labelTimestamp,
					targetId,
					label,
				};
				pathEntryIds.add(labelEntry.id);
				labelEntries.push(labelEntry);
				parentId = labelEntry.id;
			}

			this.fileEntries = [header, ...pathWithoutLabels, ...labelEntries];
			this.sessionId = newSessionId;
			this.sessionFile = newSessionFile;
			this._buildIndex();

			// 仅当其中包含 assistant 消息时才立即写入文件。
			// 否则推迟到 _persist()，由它在首次收到 assistant 回复时
			// 创建文件——这与 newSession() 的约定一致，也避免了
			// _persist() 的无 assistant 守卫后续将 flushed 重置为
			// false 时出现的头部重复问题。
			const hasAssistant = this.fileEntries.some((e) => e.type === "message" && e.message.role === "assistant");
			if (hasAssistant) {
				this._rewriteFile();
				this.flushed = true;
			} else {
				this.flushed = false;
			}

			return newSessionFile;
		}

		// 内存模式：用路径 + 标签替换当前会话
		const labelEntries: LabelEntry[] = [];
		let parentId = pathWithoutLabels[pathWithoutLabels.length - 1]?.id || null;
		for (const { targetId, label, timestamp: labelTimestamp } of labelsToWrite) {
			const labelEntry: LabelEntry = {
				type: "label",
				id: generateId(new Set([...pathEntryIds, ...labelEntries.map((e) => e.id)])),
				parentId,
				timestamp: labelTimestamp,
				targetId,
				label,
			};
			labelEntries.push(labelEntry);
			parentId = labelEntry.id;
		}
		this.fileEntries = [header, ...pathWithoutLabels, ...labelEntries];
		this.sessionId = newSessionId;
		this._buildIndex();
		return undefined;
	}

	/**
	 * 创建一个新会话。
	 * @param cwd 工作目录（存储在会话头部中）
	 * @param sessionDir 可选的会话目录。省略时使用默认目录（~/.pi/agent/sessions/<encoded-cwd>/）。
	 */
	static create(cwd: string, sessionDir?: string, options?: NewSessionOptions): SessionManager {
		const dir = sessionDir ? normalizePath(sessionDir) : getDefaultSessionDir(cwd);
		return new SessionManager(cwd, dir, undefined, true, options);
	}

	/**
	 * 打开指定的会话文件。
	 * @param path 会话文件路径
	 * @param sessionDir 可选的会话目录，用于 /new 或 /branch。省略时由文件的父目录推导。
	 * @param cwdOverride 可选的 cwd 覆盖值，替代会话头部中的 cwd。
	 */
	static open(path: string, sessionDir?: string, cwdOverride?: string): SessionManager {
		const resolvedPath = resolvePath(path);
		let header: SessionHeader | null = null;
		let preloadedFileEntries: FileEntry[] | undefined;
		if (cwdOverride === undefined && existsSync(resolvedPath)) {
			try {
				header = readSessionHeader(resolvedPath);
			} catch (error) {
				if (!(error instanceof SessionHeaderScanLimitError)) throw error;
				// 受限扫描只是发现阶段的优化。对于头部或前缀极大的历史文件，
				// 完整加载仍是权威结果。
				preloadedFileEntries = loadEntriesFromFile(resolvedPath);
				const firstEntry = preloadedFileEntries[0];
				header = firstEntry?.type === "session" ? firstEntry : null;
			}
		}
		const cwd = cwdOverride ?? (header ? getSessionHeaderCwd(header) : undefined) ?? process.cwd();
		// 若未提供 sessionDir，则从文件的父目录推导
		const dir = sessionDir ? normalizePath(sessionDir) : resolve(resolvedPath, "..");
		return new SessionManager(cwd, dir, resolvedPath, true, undefined, preloadedFileEntries);
	}

	/**
	 * 继续最近的会话；若不存在则创建新会话。
	 * @param cwd 工作目录
	 * @param sessionDir 可选的会话目录。省略时使用默认目录（~/.pi/agent/sessions/<encoded-cwd>/）。
	 */
	static continueRecent(cwd: string, sessionDir?: string): SessionManager {
		const dir = sessionDir ? normalizePath(sessionDir) : getDefaultSessionDir(cwd);
		const filterCwd = sessionDir !== undefined && dir !== getDefaultSessionDirPath(cwd);
		const mostRecent = findMostRecentSession(dir, filterCwd ? cwd : undefined);
		if (mostRecent) {
			return new SessionManager(cwd, dir, mostRecent, true);
		}
		return new SessionManager(cwd, dir, undefined, true);
	}

	/** 创建一个内存会话（不落盘），可选择基于文件系统之外提供的条目初始化。 */
	static inMemory(cwd: string = process.cwd(), options?: NewSessionOptions, entries?: FileEntry[]): SessionManager {
		return new SessionManager(cwd, "", undefined, false, options, entries);
	}

	/**
	 * 将另一个项目目录中的会话 fork 到当前项目。
	 * 在目标 cwd 中创建一个新会话，并带上源会话的完整历史。
	 * @param sourcePath 源会话文件路径
	 * @param targetCwd 目标工作目录（新会话将存放在此处）
	 * @param sessionDir 可选的会话目录。省略时使用 targetCwd 的默认目录。
	 */
	static forkFrom(
		sourcePath: string,
		targetCwd: string,
		sessionDir?: string,
		options?: NewSessionOptions,
	): SessionManager {
		const resolvedSourcePath = resolvePath(sourcePath);
		const resolvedTargetCwd = resolvePath(targetCwd);
		const sourceEntries = loadEntriesFromFile(resolvedSourcePath);
		if (sourceEntries.length === 0) {
			throw new Error(`Cannot fork: source session file is empty or invalid: ${resolvedSourcePath}`);
		}

		const sourceHeader = sourceEntries.find((e) => e.type === "session") as SessionHeader | undefined;
		if (!sourceHeader) {
			throw new Error(`Cannot fork: source session has no header: ${resolvedSourcePath}`);
		}

		const dir = sessionDir ? normalizePath(sessionDir) : getDefaultSessionDir(resolvedTargetCwd);
		if (!existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
		}

		// 创建新 ID、但内容为 fork 而来的会话文件
		if (options?.id !== undefined) {
			assertValidSessionId(options.id);
		}
		const newSessionId = options?.id ?? createSessionId();
		const timestamp = new Date().toISOString();
		const fileTimestamp = timestamp.replace(/[:.]/g, "-");
		const newSessionFile = join(dir, `${fileTimestamp}_${newSessionId}.jsonl`);

		// 写入新头部：parentSession 指向源会话，cwd 使用更新后的值
		const newHeader: SessionHeader = {
			type: "session",
			version: CURRENT_SESSION_VERSION,
			id: newSessionId,
			timestamp,
			cwd: resolvedTargetCwd,
			parentSession: resolvedSourcePath,
		};
		writeFileSync(newSessionFile, `${JSON.stringify(newHeader)}\n`, { flag: "wx" });

		// 从源会话复制所有非头部条目
		for (const entry of sourceEntries) {
			if (entry.type !== "session") {
				appendFileSync(newSessionFile, `${JSON.stringify(entry)}\n`);
			}
		}

		return new SessionManager(resolvedTargetCwd, dir, newSessionFile, true);
	}

	/**
	 * 列出某个目录下的所有会话。
	 * @param cwd 工作目录（用于计算默认会话目录）
	 * @param sessionDir 可选的会话目录。省略时使用默认目录（~/.pi/agent/sessions/<encoded-cwd>/）。
	 * @param onProgress 可选的回调，用于上报进度（已加载数, 总数）
	 */
	static async list(cwd: string, sessionDir?: string, onProgress?: SessionListProgress): Promise<SessionInfo[]> {
		const dir = sessionDir ? normalizePath(sessionDir) : getDefaultSessionDir(cwd);
		const filterCwd = sessionDir !== undefined && dir !== getDefaultSessionDirPath(cwd);
		const resolvedCwd = resolvePath(cwd);
		const sessions = (await listSessionsFromDir(dir, onProgress)).filter(
			(session) => !filterCwd || sessionCwdMatches(session.cwd, resolvedCwd),
		);
		sessions.sort((a, b) => b.modified.getTime() - a.modified.getTime());
		return sessions;
	}

	/**
	 * 列出所有项目目录下的全部会话。
	 * @param onProgress 可选的回调，用于上报进度（已加载数, 总数）
	 */
	static async listAll(onProgress?: SessionListProgress): Promise<SessionInfo[]>;
	static async listAll(sessionDir?: string, onProgress?: SessionListProgress): Promise<SessionInfo[]>;
	static async listAll(
		sessionDirOrOnProgress?: string | SessionListProgress,
		onProgress?: SessionListProgress,
	): Promise<SessionInfo[]> {
		const customSessionDir =
			typeof sessionDirOrOnProgress === "string" ? normalizePath(sessionDirOrOnProgress) : undefined;
		const progress = typeof sessionDirOrOnProgress === "function" ? sessionDirOrOnProgress : onProgress;
		if (customSessionDir) {
			const sessions = await listSessionsFromDir(customSessionDir, progress);
			sessions.sort((a, b) => b.modified.getTime() - a.modified.getTime());
			return sessions;
		}

		const sessionsDir = getSessionsDir();

		try {
			if (!existsSync(sessionsDir)) {
				return [];
			}
			const entries = await readdir(sessionsDir, { withFileTypes: true });
			const dirs = entries
				.filter((entry) => entry.isDirectory() || entry.isSymbolicLink())
				.map((entry) => join(sessionsDir, entry.name));

			// 先统计文件总数，以便进度准确
			let totalFiles = 0;
			const dirFiles: string[][] = [];
			for (const dir of dirs) {
				try {
					const files = (await readdir(dir)).filter((f) => f.endsWith(".jsonl"));
					dirFiles.push(files.map((f) => join(dir, f)));
					totalFiles += files.length;
				} catch {
					dirFiles.push([]);
				}
			}

			// 处理所有文件并跟踪进度
			let loaded = 0;
			const sessions: SessionInfo[] = [];
			const allFiles = dirFiles.flat();

			const results = await buildSessionInfosWithConcurrency(allFiles, () => {
				loaded++;
				progress?.(loaded, totalFiles);
			});

			for (const info of results) {
				if (info) {
					sessions.push(info);
				}
			}

			sessions.sort((a, b) => b.modified.getTime() - a.modified.getTime());
			return sessions;
		} catch {
			return [];
		}
	}
}
