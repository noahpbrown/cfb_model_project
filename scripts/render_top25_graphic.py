#!/usr/bin/env python

import argparse
import io
import json
import os
from typing import List, Dict, Any, Optional

import requests
from PIL import Image, ImageDraw, ImageFont


# --------------- Text helpers (Pillow 10+ safe) ---------------

def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    """
    Return (width, height) for the given text using textbbox (Pillow 10+ compatible).
    """
    if not text:
        return 0, 0
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h


# --------------- Data I/O ---------------

def load_rankings(json_path: str) -> List[Dict[str, Any]]:
    """
    Load the rankings JSON file. Expects a list of objects, each with:
      rank, team, abbr, games_played, rating_pred, spread_vs_1,
      primary_color, primary_logo_url, wins_cum, losses_cum, etc.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Ensure sorted by rank
    data = sorted(data, key=lambda x: x.get("rank", 9999))
    return data


def download_logo(url: str, size: int = 96) -> Optional[Image.Image]:
    """
    Download a team logo from URL and resize to (size, size).
    Returns a PIL Image, or None if download fails.
    """
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception:
        return None


# --------------- Rendering ---------------

def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    Try a couple of fonts; fall back to PIL default if none found.
    """
    candidates = [
        "/System/Library/Fonts/SFNSRounded.ttf",  # macOS example
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for path in candidates:
        try:
            if path and os.path.exists(path):
                return ImageFont.truetype(path, size=size)
        except Exception:
            continue

    # Fallback
    return ImageFont.load_default()


def render_top25(rankings: List[Dict[str, Any]], out_path: str) -> None:
    # Canvas size (portrait)
    width, height = 1080, 1920
    bg_color = (8, 8, 12)  # dark background

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Fonts
    font_header = load_font(72, bold=True)
    font_sub = load_font(32, bold=False)
    font_rank = load_font(40, bold=True)
    font_team = load_font(36, bold=True)
    font_record = load_font(28, bold=False)
    font_rating = load_font(32, bold=False)
    font_spread = load_font(36, bold=True)
    font_col_header = load_font(28, bold=True)

    # --- Header ---
    header_text = "NOAHDAWG34 POLL"
    sub_text = "Week Fourteen Edition"

    header_y = 40

    w_header, h_header = text_size(draw, header_text, font_header)
    draw.text(
        ((width - w_header) // 2, header_y),
        header_text,
        font=font_header,
        fill="white",
    )

    w_sub, h_sub = text_size(draw, sub_text, font_sub)
    draw.text(
        ((width - w_sub) // 2, header_y + h_header + 10),
        sub_text,
        font=font_sub,
        fill="#ff6666",
    )

    # Line under header
    line_y = header_y + h_header + h_sub + 30
    draw.line(
        [(80, line_y), (width - 80, line_y)],
        fill=(60, 60, 80),
        width=3,
    )

    # --- List area layout ---
    top_margin = line_y + 60  # leave room for column headers
    bottom_margin = 60
    available_height = height - top_margin - bottom_margin

    max_rows = min(25, len(rankings))
    row_height = available_height // max_rows

    # horizontal layout
    left_margin = 80
    right_margin = 80

    # columns: rank | logo | team text | rating | spread
    rank_col_x = left_margin
    logo_col_x = rank_col_x + 70
    logo_size = 72

    text_start_x = logo_col_x + logo_size + 24

    # Fixed right edges for rating & spread columns
    spread_col_right = width - right_margin
    rating_spread_gap = 150  # horizontal gap between rating and spread columns
    rating_col_right = spread_col_right - rating_spread_gap

    # --- Column headers ("Rating" and "Spread") ---
    header_row_y = top_margin - 40  # just above first data row

    rating_label = "Rating"
    spread_label = "Spread"

    rating_label_w, rating_label_h = text_size(draw, rating_label, font_col_header)
    spread_label_w, spread_label_h = text_size(draw, spread_label, font_col_header)

    draw.text(
        (rating_col_right - rating_label_w, header_row_y),
        rating_label,
        font=font_col_header,
        fill="#bbbbbb",
    )
    draw.text(
        (spread_col_right - spread_label_w, header_row_y),
        spread_label,
        font=font_col_header,
        fill="#bbbbbb",
    )

    # Pre-download logos
    logos_cache: Dict[int, Optional[Image.Image]] = {}
    for row in rankings[:max_rows]:
        team_id = row.get("team_id")
        logo_url = row.get("primary_logo_url")
        if team_id not in logos_cache:
            logos_cache[team_id] = download_logo(logo_url, size=logo_size)

    # --- Draw each row ---
    for idx, row in enumerate(rankings[:max_rows]):
        y_top = top_margin + idx * row_height
        y_bottom = y_top + row_height

        # Slight alternating row shading
        row_bg = (15, 15, 25) if idx % 2 == 0 else (20, 20, 30)
        draw.rectangle([(40, y_top + 4), (width - 40, y_bottom - 4)], fill=row_bg)

        rank = row.get("rank")
        team_name = row.get("team")
        abbr = row.get("abbr", "")
        wins = row.get("wins_cum")
        losses = row.get("losses_cum")
        rating = float(row.get("rating_pred", 0.0))
        spread_raw = float(row.get("spread_vs_1", 0.0))
        spread = round(spread_raw * 2) / 2  # round to nearest 0.5
        color = row.get("primary_color", "#ffffff")

        # rank
        rank_text = str(rank)
        rank_y_center = (y_top + y_bottom) // 2
        rank_w, rank_h = text_size(draw, rank_text, font_rank)
        draw.text(
            (rank_col_x, rank_y_center - rank_h // 2),
            rank_text,
            font=font_rank,
            fill="#ffffff",
        )

        # logo
        team_id = row.get("team_id")
        logo_img = logos_cache.get(team_id)
        logo_y = rank_y_center - logo_size // 2

        if logo_img is not None:
            img.paste(logo_img, (logo_col_x, logo_y), logo_img)
        else:
            # fallback: colored circle
            try:
                rgb = tuple(
                    int(color.strip("#")[i: i + 2], 16) for i in (0, 2, 4)
                )
            except Exception:
                rgb = (200, 200, 200)
            r = logo_size // 2
            cx = logo_col_x + r
            cy = rank_y_center
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)],
                fill=rgb,
            )

        # team name
        team_y_center = rank_y_center
        team_w, team_h = text_size(draw, team_name, font_team)
        team_y = team_y_center - team_h

        draw.text(
            (text_start_x, team_y),
            team_name,
            font=font_team,
            fill="#ffffff",
        )

        # record (W-L) under team name; fallback to abbr if needed
        if isinstance(wins, (int, float)) and isinstance(losses, (int, float)):
            rec_text = f"{int(round(wins))}-{int(round(losses))}"
        else:
            rec_text = abbr.upper()

        rec_w, rec_h = text_size(draw, rec_text, font_record)
        draw.text(
            (text_start_x, team_y + team_h + 4),
            rec_text,
            font=font_record,
            fill="#aaaaaa",
        )

        # spread label (using rounded spread)
        if abs(spread_raw) < 1e-6:
            spread_text = "--"
        else:
            spread_text = f"+{spread:.1f}"

        spread_w, spread_h = text_size(draw, spread_text, font_spread)
        spread_x = spread_col_right - spread_w
        spread_y = rank_y_center - spread_h // 2

        # rating text
        rating_text = f"{rating:.1f}"
        rating_w, rating_h = text_size(draw, rating_text, font_rating)
        rating_x = rating_col_right - rating_w
        rating_y = rank_y_center - rating_h // 2

        # draw rating (light gray)
        draw.text(
            (rating_x, rating_y),
            rating_text,
            font=font_rating,
            fill="#cccccc",
        )

        # draw spread (single consistent color)
        spread_fill = "#ffcc66"  # gold-ish for all non-top teams
        draw.text(
            (spread_x, spread_y),
            spread_text,
            font=font_spread,
            fill=spread_fill if spread_text != "--" else "#ffffff",
        )

    # --- Footer / credit ---
    footer_text = "@noahdawg34 Model"
    footer_font = load_font(26, bold=False)
    w_footer, h_footer = text_size(draw, footer_text, footer_font)
    draw.text(
        ((width - w_footer) // 2, height - h_footer - 20),
        footer_text,
        font=footer_font,
        fill="#888888",
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    print(f"Saved graphic to {out_path}")


# --------------- CLI ---------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render Top 25 graphic from model output JSON."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to rankings JSON file (e.g., outputs/top25_2025_week14.json)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to output PNG (e.g., outputs/top25_2025_week14.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rankings = load_rankings(args.json_path)
    print(f"Loaded {len(rankings)} teams from {args.json_path}")
    render_top25(rankings, out_path=args.out_path)


if __name__ == "__main__":
    main()
