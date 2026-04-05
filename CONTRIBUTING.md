# Contributing

Thank you for your interest in contributing! This guide will help you get
started with the development workflow.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Development Setup](#development-setup)
- [Making Changes](#making-changes)
  - [Branching Strategy](#branching-strategy)
  - [Code Style](#code-style)
  - [Commit Messages](#commit-messages)
  - [Running Tests](#running-tests)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)
- [Project Architecture](#project-architecture)
- [Legal](#legal)

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| [Java](https://adoptium.net/) | **21+** | Required (LTS recommended) |
| [Gradle](https://gradle.org/) | **9.4+** | Or use the included `./gradlew` wrapper |
| [Git](https://git-scm.com/) | **2.x+** | For cloning and version control |

### Development Setup

1. **Fork** the repository on GitHub.

2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/<your-username>/<repo>.git
   cd <repo>
   ```

3. **Build** the project to verify everything works:

   ```bash
   ./gradlew build
   ```

> [!TIP]
> You do not need to install Gradle separately. The `gradlew` wrapper script
> in the repository root will download the correct version automatically.

## Making Changes

### Branching Strategy

All development is based on the `master` branch.

1. Create a feature branch from `master`:

   ```bash
   git checkout -b feature/your-feature master
   ```

2. Make your changes in small, focused commits.

3. Push your branch and open a pull request against `master`.

### Code Style

- Follow standard Java conventions and the existing patterns in the codebase.
- Use [Lombok](https://projectlombok.org/) annotations where the project
  already does (e.g., `@Getter`, `@RequiredArgsConstructor`).
- Use [JetBrains annotations](https://github.com/JetBrains/java-annotations)
  (`@NotNull`, `@Nullable`) for nullability contracts.
- Keep methods short and focused. Prefer clear names over comments.

### Commit Messages

Write commit messages in **imperative mood** (e.g., "Add support for X", not
"Added support for X" or "Adds support for X").

```
Add PNG alpha channel support

Implement alpha channel handling in PngImageReader to support
RGBA pixel buffers when the source image uses transparency.
```

- First line: concise summary, 72 characters or fewer.
- Blank line, then an optional body with additional context.

### Running Tests

```bash
# Run all tests
./gradlew test

# Run a specific test class
./gradlew test --tests "dev.simplified.SomeTest"

# Full build (compile + test + checks)
./gradlew build
```

> [!NOTE]
> All pull requests must pass the existing test suite. If your change adds new
> behavior, include corresponding tests.

## Submitting a Pull Request

1. Ensure your branch is up to date with `master`:

   ```bash
   git fetch origin
   git rebase origin/master
   ```

2. Run the full build locally:

   ```bash
   ./gradlew build
   ```

3. Push your branch and open a pull request on GitHub.

4. In the PR description, explain **what** your change does and **why**.

5. A maintainer will review your PR and may request changes before merging.

## Reporting Issues

- Use [GitHub Issues](../../issues) to report bugs or request features.
- Include steps to reproduce, expected behavior, and actual behavior.
- Mention your Java version and operating system.

## Project Architecture

This is a single-module Gradle project built with `java-library`. Sources live
under `src/main/java` and tests under `src/test/java`. The project publishes to
[JitPack](https://jitpack.io/) directly from the `master` branch.

Key conventions:

- **Package root** - `dev.simplified.*` under the group ID `dev.simplified`.
- **Dependencies** - Other Simplified-Dev modules are pulled from JitPack as
  `com.github.simplified-dev:<artifact>:master-SNAPSHOT`.
- **Java 21** - The toolchain is locked to Java 21 via `build.gradle.kts`.

## Legal

By submitting a contribution, you agree that your work will be licensed under
the [Apache License 2.0](LICENSE.md), the same license that covers this project.
