from __future__ import annotations

import random
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Self

from uiregress.data.augment import add_subtle_pixel_noise
from uiregress.data.mutations import choose_mutation, no_regression_mutation
from uiregress.data.schema import BoundingBox, MutationSpec

try:
    from playwright.async_api import Browser, Page, Playwright, async_playwright
except ImportError:  # pragma: no cover - exercised on installations without dataset extra
    Browser = Any  # type: ignore[misc,assignment]
    Page = Any  # type: ignore[misc,assignment]
    Playwright = Any  # type: ignore[misc,assignment]
    async_playwright = None


@dataclass(frozen=True, slots=True)
class RenderedPair:
    label: str
    mutation: MutationSpec
    region: BoundingBox | None
    browser: dict[str, str]


BROWSER_LAUNCH_TIMEOUT_MS = 15_000
PAGE_ACTION_TIMEOUT_MS = 10_000


class PlaywrightRenderer:
    def __init__(self, *, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Viewport dimensions must be positive.")
        self.viewport = {"width": width, "height": height}
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def __aenter__(self) -> Self:
        if async_playwright is None:
            raise RuntimeError(
                "Playwright is not installed. Run `uv sync --all-extras` and "
                "`uv run playwright install chromium`."
            )

        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                channel="chromium",
                timeout=BROWSER_LAUNCH_TIMEOUT_MS,
            )
        except Exception as exc:
            await self._playwright.stop()
            self._playwright = None
            raise RuntimeError(
                "Chromium could not be launched with Playwright. "
                "Run `uv run playwright install chromium`. "
                f"Original error: {exc}"
            ) from exc
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._browser = None
        self._playwright = None

    async def _page(self) -> Page:
        if self._browser is None:
            raise RuntimeError("PlaywrightRenderer must be used as an async context manager.")
        context = await self._browser.new_context(
            viewport=self.viewport,
            device_scale_factor=1,
            locale="en-US",
            color_scheme="light",
            reduced_motion="reduce",
        )
        page = await context.new_page()
        page.set_default_timeout(PAGE_ACTION_TIMEOUT_MS)
        page.set_default_navigation_timeout(PAGE_ACTION_TIMEOUT_MS)
        await page.emulate_media(reduced_motion="reduce", color_scheme="light")
        return page

    @staticmethod
    async def _selectors(page: Page) -> list[str]:
        target_names = await page.locator("[data-uiregress-target]").evaluate_all(
            "els => els.map(el => el.dataset.uiregressTarget)"
        )
        unique_names = sorted({str(name) for name in target_names if name})
        return [f'[data-uiregress-target="{name}"]' for name in unique_names]

    @staticmethod
    async def _apply_mutation(page: Page, mutation: MutationSpec) -> None:
        if mutation.selector is None:
            return

        applied = await page.evaluate(
            """
            ({selector, operator, parameters}) => {
              const element = document.querySelector(selector);
              if (!element) return false;

              const style = element.style;
              switch (operator) {
                case "hide_element":
                  style.visibility = "hidden";
                  break;
                case "translate_element":
                  style.transform = `translate(${parameters.dx}px, ${parameters.dy}px)`;
                  style.position = "relative";
                  style.zIndex = "10";
                  break;
                case "increase_width":
                  style.width = `${element.getBoundingClientRect().width * parameters.scale}px`;
                  style.maxWidth = "none";
                  style.position = "relative";
                  style.zIndex = "10";
                  break;
                case "overflow_hidden":
                  const clippedHeight = Math.max(
                    8,
                    element.getBoundingClientRect().height * parameters.height_ratio
                  );
                  style.height = `${clippedHeight}px`;
                  style.overflow = "hidden";
                  break;
                case "change_style_token":
                  style.background = parameters.background;
                  style.color = parameters.color;
                  break;
                default:
                  throw new Error(`Unknown mutation operator: ${operator}`);
              }
              return true;
            }
            """,
            {
                "selector": mutation.selector,
                "operator": mutation.operator,
                "parameters": mutation.parameters,
            },
        )
        if not applied:
            raise RuntimeError(f"Mutation target disappeared: {mutation.selector}")

    @staticmethod
    def _playwright_version() -> str:
        try:
            return version("playwright")
        except PackageNotFoundError:
            return "unknown"

    async def render_pair(
        self,
        fixture: Path,
        baseline_path: Path,
        current_path: Path,
        *,
        rng: random.Random,
        sample_seed: int,
        no_regression: bool,
    ) -> RenderedPair:
        if not fixture.is_file():
            raise FileNotFoundError(f"Fixture not found: {fixture}")

        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.parent.mkdir(parents=True, exist_ok=True)

        page = await self._page()
        try:
            await page.goto(
                fixture.resolve().as_uri(),
                wait_until="domcontentloaded",
                timeout=PAGE_ACTION_TIMEOUT_MS,
            )
            await page.screenshot(
                path=str(baseline_path),
                full_page=False,
                animations="disabled",
            )

            if no_regression:
                mutation = no_regression_mutation()
                add_subtle_pixel_noise(
                    baseline_path,
                    current_path,
                    seed=sample_seed,
                    amplitude=int(mutation.parameters["amplitude"]),
                )
                region = None
            else:
                selectors = await self._selectors(page)
                mutation = choose_mutation(rng, selectors)
                locator = page.locator(mutation.selector)
                before = BoundingBox.from_mapping(await locator.bounding_box())
                if before is None:
                    raise RuntimeError(f"Mutation target is not visible: {mutation.selector}")

                await self._apply_mutation(page, mutation)
                await page.wait_for_timeout(50)
                after = BoundingBox.from_mapping(await locator.bounding_box())
                region = before.union(after)
                await page.screenshot(
                    path=str(current_path),
                    full_page=False,
                    animations="disabled",
                )

            browser_version = self._browser.version if self._browser is not None else "unknown"
            return RenderedPair(
                label=mutation.label,
                mutation=mutation,
                region=region,
                browser={
                    "engine": "chromium",
                    "browser_version": browser_version,
                    "playwright_version": self._playwright_version(),
                },
            )
        finally:
            await page.context.close()
