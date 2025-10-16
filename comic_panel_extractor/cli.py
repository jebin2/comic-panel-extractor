#!/usr/bin/env python3
"""
Command-line interface for comic panel extraction.
"""

import argparse
import sys
import json
from typing import Optional, List

from .main import ComicPanelExtractor
from .config import Config, load_config


class ComicPanelCLI:
	"""Command-line interface for comic panel extraction."""
	
	def __init__(self):
		self.parser = self._create_parser()
	
	def _create_parser(self) -> argparse.ArgumentParser:
		"""Create argument parser."""
		parser = argparse.ArgumentParser(
			prog="comic-extract",
			description="Extract panels from comic book images using OCR and image processing",
			formatter_class=argparse.RawDescriptionHelpFormatter,
			epilog="""
Examples:
  comic-extract comic.jpg
  comic-extract comic.jpg --config config.json
			"""
		)
		
		# Required arguments
		parser.add_argument(
			"input_path",
			help="Path to the comic image file"
		)
		
		# Configuration file
		parser.add_argument(
			"--config",
			help="Path to JSON configuration file"
		)
		
		return parser
	
	def run(self, args: Optional[List[str]] = None) -> int:
		"""Main CLI entry point."""
		try:
			parsed_args = self.parser.parse_args(args)
			# Load configuration
			config = self._load_config(parsed_args)
			ComicPanelExtractor(config).extract_panels_from_comic()
		except Exception as e:
			print(f"❌ Error: {e}", file=sys.stderr)
			return 1
	
	def _load_config(self, args: argparse.Namespace) -> Config:
		"""Load configuration from file or create from arguments."""
		config = load_config()
		
		# Load from config file if provided
		if args.config:
			try:
				with open(args.config, 'r', encoding='utf-8') as f:
					config_data = json.load(f)
					for key, value in config_data.items():
						if hasattr(config, key):
							setattr(config, key, value)
			except Exception as e:
				print(f"⚠️  Warning: Could not load config file: {e}", file=sys.stderr)
		
		# Override with command line arguments
		config.input_path = args.input_path
		
		return config

def main():
	"""Main entry point for CLI."""
	cli = ComicPanelCLI()
	sys.exit(cli.run())


if __name__ == "__main__":
	main()