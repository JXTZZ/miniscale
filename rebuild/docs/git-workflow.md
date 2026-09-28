# 阅读分支差异、推送与合并

截至 2026-09-28，本地分支是 `learn/rebuild-from-scratch`，重构提交是 `a2e8630`；
本地与远端 main 均为 `6b485e4`，远端尚没有重构分支。本文是操作说明，本轮没有执行 push 或 merge。

## 三个动作分别做什么

- `commit`：把当前改动保存为本地历史记录。
- `push`：把本地分支上传到 GitHub，供备份、浏览和创建 PR。
- `merge`：把一个分支的改动纳入另一个分支。

推送 `learn/rebuild-from-scratch` 后仍可继续阅读、修改；它不会自动合并进 main。
当前分支主要是新增 `rebuild/`，所以将来合并也会把这个目录带入 main，
**不会自动把 `rebuild/src/` 替换成根目录 `src/`**。把 rebuild 提升为唯一主项目需要另做一次路径迁移。

## 1. 先比较，暂时不切换分支

从仓库根目录运行：

```bash
cd /home/lotusy/hy/miniscale
git status --short --branch
git log --oneline --decorate --graph --all -12
git diff --stat main...learn/rebuild-from-scratch
git diff --name-status main...learn/rebuild-from-scratch
```

`main...分支名` 显示自共同起点以来该分支提交的改动。`git diff` 查看尚未暂存的改动，
`git diff --cached` 查看已暂存的改动，`git status` 还会列出新文件。

因为新代码在新增目录中，Git 的分支 diff 不会自动把两套实现逐文件并排匹配。重点这样看：

```bash
git diff --no-index src/miniscale/model.py rebuild/src/miniscale/model.py
git diff --no-index src/miniscale/pipeline.py rebuild/src/miniscale/pipeline.py
git diff --no-index src/miniscale/data/__init__.py rebuild/src/miniscale/data/pretrain.py
git diff --no-index src/miniscale/training/stages/pretrain.py rebuild/src/miniscale/training/pretrain/runner.py
```

前两项当前没有差异；后两项同时包含文件拆分和实现位置变化。`--no-index` 遇到差异会返回退出码 1，属于正常结果。
配合 [检查报告](rebuild-review.md) 的路径映射阅读。

## 2. 配置提交身份与登录

Git 姓名/邮箱决定提交署名，GitHub 登录决定是否有推送权限，是两项独立配置。
上一条提交使用了命令级身份参数；如果希望本仓库以后都使用同一身份：

```bash
git config --local user.name JXTZZ
git config --local user.email 2817720265@qq.com
```

之前 HTTPS push 缺少凭据；GitHub 连接器写入也返回 403。可以在自己的终端安装 GitHub CLI，
按 [官方安装说明](https://github.com/cli/cli#installation) 安装 `gh` 后执行：

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

按提示在浏览器登录并授权。`gh auth login` 建立登录，`gh auth setup-git` 让 Git 使用该凭据。
参见 [login 文档](https://cli.github.com/manual/gh_auth_login) 和 [setup-git 文档](https://cli.github.com/manual/gh_auth_setup-git)。
不要把密码或 token 写进仓库或聊天。

## 3. 阅读完成后提交本轮说明，再推送分支

本轮新增的下载脚本和说明需要单独提交。下面只暂存这些文件：

```bash
git add rebuild/README.md rebuild/scripts/download_minimind.py \
  rebuild/docs/rebuild-review.md rebuild/docs/learning-guide.md rebuild/docs/git-workflow.md
git diff --cached --stat
git diff --cached --check
git commit -m "docs: audit rebuild equivalence and document learning workflow"
git push -u origin learn/rebuild-from-scratch
```

`-u` 建立本地与远端分支的跟踪关系，之后在这个分支上通常直接 `git push` 即可。
下载的数据和 `artifacts/` 已被忽略；提交前查看 `git status` 确认范围。

推送后验证：

```bash
git fetch origin
git status --short --branch
git rev-parse HEAD
git rev-parse origin/learn/rebuild-from-scratch
```

最后两行哈希应相同。推送说明见 [git push 文档](https://git-scm.com/docs/git-push)。

## 4. 之后合并：建议先用 GitHub PR 阅读差异

完成推送后，在仓库页面创建 Pull Request：

- base：`main`
- compare：`learn/rebuild-from-scratch`
- 在 Files changed 中阅读改动，确认测试和项目目录安排。
- 尚未决定合并时，可创建 Draft PR 并保留。

确认以后再点击 Merge。完成后本地用 `git switch main`、`git pull --ff-only origin main` 同步。

也可以将来在本地合并。**以下是未来操作示例，本轮没有执行：**

```bash
git switch main
git pull --ff-only origin main
git merge --no-ff --no-commit learn/rebuild-from-scratch
```

`--no-ff --no-commit` 会停在创建合并提交之前，便于检查暂存改动和重新跑测试。
如果有冲突，先逐个解决；若暂不合并，可执行 `git merge --abort`。
确认后才执行：

```bash
git commit -m "merge: add rebuild training project"
git push origin main
```

执行合并前先让工作区干净。详细语义见 [git merge 文档](https://git-scm.com/docs/git-merge)。
