# 供应链固定与细粒度沙箱（两道管控）

> 状态（2026-10-09）：
> **供应链固定（§1）已落地并已实测**；
> **细粒度沙箱档（§2）配置可写、CLI 也认，但本机 Windows 后端兑现不了其中的保证**
> ——实测记录见 §2.5，结论与替代方案见 §2.6。**本机暂不启用**。

本课题的「真机复跑」依赖从外网拉代码与二进制（PJC 源码、bazelisk、
Bazel 的第三方依赖、pip/conda 包）。控件只有两个抓手：
**拉什么（钉死版本）** 与 **能拉到哪（域名白名单）**。

---

## 1. 供应链固定（已落地，已实测）

### 1.1 钉死了什么

| 对象 | 固定值 | 校验方式 |
|---|---|---|
| 上游源码 `google/private-join-and-compute` | commit `950c5e4c88d7effe85147beb7856152f7c53394b`（2026-03-09） | `git rev-parse HEAD` 断言相等，不等即 `die` |
| `bazelisk`（Bazel 启动器） | `v1.29.0` 的 `bazelisk-linux-amd64` | `sha256sum -c`，期望 `5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992` |
| `openmined-psi`（PSI-CA 依赖，Phase 11） | `2.0.6` 的 manylinux 轮子（cp310 / cp311 × manylinux_2_35 / _2_39） | `pip install --require-hashes`（清单 `requirements-psi-ca.txt`），哈希不符即失败；实测正反例见 `docs/PSI_CA_CAPABILITY.md` §8.1 |
| `protobuf`（同上依赖） | `6.30.2`（manylinux2014_x86_64 / py3-none-any） | 同上（同一个哈希固定清单） |

改动全在 `scripts/verify_pi_sum_wsl.sh`：不再用 `releases/latest`、
不再 `clone` 默认分支最新；两者都可以用同名环境变量覆盖
（`PJC_GIT_PIN` / `BAZELISK_VERSION` / `BAZELISK_SHA256`），默认值即上表。

### 1.2 实测证据（2026-10-09，WSL2）

```text
== 1) bazelisk sha256：正例（固定版本下载物） ==
/tmp/pjc_setup/bin/bazelisk: OK
== 2) bazelisk sha256：反例（篡改 1 字节） ==
结果: 已拒绝（符合预期）
== 3) 上游按 sha fetch：HEAD 是否等于钉死值 ==
HEAD   = 950c5e4c88d7effe85147beb7856152f7c53394b
EXPECT = 950c5e4c88d7effe85147beb7856152f7c53394b
结果: 一致
日期       = 2026-03-09
.bazelversion = 8.0.1
```

第 3 项同时确认 GitHub 支持「按 sha fetch」这条路径；`.bazelversion = 8.0.1`
与 `docs/PSI_SUM_CAPABILITY.md` §8 登记的实测一致。

### 1.3 还没固定的（如实登记）

- **Bazel 自己拉的第三方依赖**（absl / grpc / protobuf…）：由上游 `WORKSPACE`
  的 `http_archive` 声明与校验，本仓库不改上游源码，这一层不在本项目控制内。
- **pip / conda 依赖**：`requirements-jax.txt` / `requirements-spu.txt` 只钉版本号，
  **没做 hash pinning**；PSI-CA 档已闭合这一项（`requirements-psi-ca.txt`，
  Phase 11，`--require-hashes` + 正反例实测，见 `docs/PSI_CA_CAPABILITY.md` §8.1）。
- **上游 commit 无签名**：GitHub API 显示该 commit `verified: false`（unsigned）。
  钉死只防「上游漂移」，不防「上游投毒」。
- **SPU wheel**：来源与版本见 `requirements-spu.txt` 与
  `docs/VERSION_COMPATIBILITY.md`，同样只有版本号。

### 1.4 怎么复跑

```bash
bash scripts/verify_pi_sum_wsl.sh
```

要换固定值就覆盖环境变量（例如 `PJC_GIT_PIN=<sha> bash scripts/verify_pi_sum_wsl.sh`），
换完记得同步本文件与 `docs/PSI_SUM_CAPABILITY.md`。

---

## 2. 细粒度沙箱档（已实测：配置可写、CLI 认，但本机后端兑现不了）

### 2.1 为什么值得做

本机 Codex 现在跑命令用的是 `danger-full-access`（无隔离）：任何一条命令，
包括 `pip install` / `curl | bash` 拉回来的东西，都能读整个磁盘、写任何位置、
连任何地址。本课题要从外网拉源码与二进制，这正是主要风险面。

### 2.2 官方规则要点（已核对，出处见文末）

- **与旧设置互斥**：`default_permissions` + `[permissions]` 与
  `sandbox_mode` / `[sandbox_workspace_write]` 只能二选一；**只要 `sandbox_mode`
  出现在任何已加载的配置文件里，Codex 就走旧沙箱设置，档不生效**。
- **网络是三个开关联动**：

  | 网络 | 代理 `features.network_proxy` | 结果 |
  | --- | --- | --- |
  | 关 | 任意 | 命令完全不能出网 |
  | 开 | 关 | 直连、不受限（域名白名单**不生效**） |
  | 开 | 开 | 走代理，按域名白名单放行 |

- **域名规则是 allowlist-first**：精确主机只匹配自己；`*.example.com` 只匹配子域；
  `**.example.com` 匹配顶点 + 子域；`*` 是「全局放行」且**只能用于 allow**；
  `deny` 永远优先于 `allow`；白名单为空时，代理挡掉所有外部目的地。
- **本机 / 内网默认被挡**（防 DNS rebinding）：要放行本机需显式写
  `localhost` / `127.0.0.1`，或设 `allow_local_binding = true`。
- **文件系统**：条目值 `read` / `write` / `deny`；越具体越优先；`deny > write > read`；
  路径形式有 `:minimal` / `:workspace_roots` / `~/path` 等，支持 `"**/*.env"` 这类 glob。
- **边界（官方明说）**：命令网络代理**只管沙箱内的本地命令流量**，**不管**
  web search / Apps 与连接器 / MCP 服务器 / 浏览器与 Computer Use /
  **Codex 服务流量（模型、认证）** / Codex Cloud——「命令网络白名单不是全局网络策略」。
- CLI 侧工具：`codex sandbox -P <档名> -C <目录> -- <命令>`
  （本机 `codex sandbox` 就是 Windows 后端；Linux/WSL 上另有 `codex sandbox linux`）。

### 2.3 本档内容

见 `scripts/codex_permissions.example.toml`（可直接并入 `config.toml`）：
文件系统收成「工作区可读写 + `:minimal`」，并把 `**/*.env` 挖成 `deny`；
出网只放行本课题真正要用的域名（GitHub / Bazel 分发 / PyPI / 阿里云与清华镜像 /
conda），未列出的一律挡掉。

### 2.4 应用步骤（等 §2.6 的前置条件满足再做）

1. 备份 `%USERPROFILE%\.codex\config.toml`；
2. 删掉 `sandbox_mode = "..."` 与 `[sandbox_workspace_write]`（否则本档不生效）；
3. 并入 §2.3 的内容；
4. 重启 Codex。

先试再定：**别改 `default_permissions`**，用临时 `CODEX_HOME`（或 `-p <档名>` 分层）
配 `codex sandbox -P <档名> -C <目录> -- <命令>` 试跑，确认无误再切默认。

### 2.5 实测记录（2026-10-09，本机 Windows，Codex CLI `0.162.0-alpha.2`）

测法：把档写进**临时 `CODEX_HOME`**（不动现有 `config.toml`），用
`codex sandbox -P <档名> -C <目录> -- <命令>` 跑探针。CLI 版本 ≥ 0.138.0，
官方「支持权限档」的前置条件满足。

| # | 输入 | 结果 |
|---|---|---|
| 1 | `-P gis-spu-noglob`（去掉 deny-glob 的变体） | 沙箱起得来，命令正常执行（`PROBE_START`/`PROBE_END`，exit 0）——**说明档确实被解析并应用** |
| 2 | `-P gis-spu`（带 `**/*.env` deny-glob） | `windows sandbox failed: Restricted read-only access requires the elevated Windows sandbox backend` |
| 3 | 同 2，但工作目录落在仓库父目录 | 报 `failed to run bundled ripgrep for unreadable glob scan under <目录>: 拒绝访问。 (os error 5)`（该树里有 ACL 坏掉的 `GIS_SPU/pytest-of-DELL/`） |
| 4 | 档内 HTTPS（curl / schannel） | 白名单内外**都**失败：`curl: (35) schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS` |
| 5 | 档内 HTTPS（Python / OpenSSL） | `pypi.org` = 200；**`example.com` = 200（白名单外也通）** |
| 6 | 档内环境变量 | `HTTP_PROXY=[] HTTPS_PROXY=[] ALL_PROXY=[]`（未注入代理） |
| 7 | 档内文件系统 | 读工作区外 `C:\Users\DELL\.codex\config.toml` **成功**；写工作区外被拒（符合预期）；**写工作区内 `codextest2\inside_write.txt` 也被拒**（不符合预期） |
| 8 | `codex sandbox windows --help` | `windows sandbox failed: helper_unknown_error: setup refresh had errors` |

旁注：跑完这轮沙箱探针后，本会话连续出现两次 shell 层失败
（`Failed to create unified exec process: 拒绝访问。 (os error 5)` 与一次 `aborted`），
数十秒后自行恢复——沙箱 helper 的 "setup refresh" 会改机器状态
（沙箱用户 / 防火墙规则）。**建议不要在主力机器上反复试跑。**

### 2.6 结论与取舍（2026-10-09）

**做得到的**：档写得出来、CLI 认、`-P` 确实生效（§2.5 第 1 行）；
供应链固定已落地并实测（§1）。

**做不到的（本机现状）**：
- 「受限只读」（deny-glob）要求 elevated 后端，本机起不来（第 2、8 行）；
- 即便沙箱起来了，保证也兑现不了：白名单没拦住 `example.com`（第 5、6 行）、
  工作区内写入被误拦、工作区外读取没拦住（第 7 行）；
- 沙箱内 schannel TLS 直接不可用（第 4 行）→ `curl`、Git for Windows 这类
  走 schannel 的工具在档内不能出网。

**因此本期建议**：
1. `config.toml` **维持现状**（`danger-full-access`），现在**不要**切档；
2. 真要拿细粒度沙箱，走**在 WSL 里跑 Codex（Linux 后端）**这条路——官方在
   Linux/WSL 用的是 bubblewrap + seccomp/landlock，正是文档描述的那套；
   Windows 后端（`elevated` / `mxc` / `unelevated`）本次实测达不到同等效果；
3. 本机当前的管控靠另外三条：§1 的供应链固定、真机复跑改在**Codex 之外的终端**
   执行（`scripts/verify_pi_sum_wsl.sh` 本就是独立脚本）、只在本仓库内做可控动作。

### 2.7 待办（下期）

- 在 WSL 里装 Linux 版 Codex CLI，用同一个档重跑 §2.5 的第 1 / 4 / 5 / 7 项，
  看 Linux 后端能否兑现——**这是本档能否启用的分水岭**。
- 若要继续用 Windows 后端：先查 `helper_unknown_error: setup refresh had errors`
  （与 `[windows] sandbox = "elevated"` 相关），再解决 deny-glob 的后端要求。
- 清掉 `GIS_SPU/pytest-of-DELL/` 的坏 ACL（它会连带打断沙箱的 glob 扫描，
  见 §2.5 第 3 行）；该目录此前已识别，待手工删除。
- 未验证：`:minimal` 是否覆盖 `~\.codex`——若覆盖，§2.5 第 7 行的「读成功」不算越权。

---

**规范来源**：Codex 手册 `https://developers.openai.com/codex/codex-manual.md`
（本机缓存 `work/tmp/codex-manual.md`，
sha256 `126b0e855df8587551bfdb6273be6134a2417c0391fe58be6d928a3e31abe680`，
与官方响应头 `X-Content-Sha256` 一致）。§2.2 的规则要点均为该文档原文口径。
