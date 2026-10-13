"""
Tests for logger utility module.

Tests logger setup, configuration, and utility functions.
"""

import pytest
import logging
from src.utils.logger import setup_logger


class TestLoggerSetup:
    """Test logger setup functionality."""
    
    @pytest.mark.unit
    def test_setup_logger_creates_logger(self):
        """Test that setup_logger creates a logger."""
        logger = setup_logger('test_logger')
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_logger'
    
    @pytest.mark.unit
    def test_setup_logger_default_level(self):
        """Test default log level is INFO."""
        logger = setup_logger('test_default_level')
        
        assert logger.level == logging.INFO
    
    @pytest.mark.unit
    def test_setup_logger_custom_level(self):
        """Test setting custom log level."""
        logger = setup_logger('test_custom_level', level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG
    
    @pytest.mark.unit
    def test_setup_logger_has_handler(self):
        """Test logger has stream handler."""
        logger = setup_logger('test_handler')
        
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    @pytest.mark.unit
    def test_setup_logger_idempotent(self):
        """Test calling setup_logger twice doesn't duplicate handlers."""
        logger1 = setup_logger('test_idempotent')
        handler_count1 = len(logger1.handlers)
        
        logger2 = setup_logger('test_idempotent')
        handler_count2 = len(logger2.handlers)
        
        assert handler_count1 == handler_count2
        assert logger1 is logger2
    
    @pytest.mark.unit
    def test_logger_can_log(self):
        """Test logger can log messages."""
        logger = setup_logger('test_can_log')
        
        # Should not raise any exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")


class TestLoggerIntegration:
    """Integration tests for logger module."""
    
    @pytest.mark.integration
    def test_multiple_loggers_independent(self):
        """Test multiple loggers operate independently."""
        logger1 = setup_logger('test_logger_1', level=logging.DEBUG)
        logger2 = setup_logger('test_logger_2', level=logging.WARNING)
        
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.WARNING
        assert logger1 is not logger2
    
    @pytest.mark.integration
    def test_logger_hierarchy(self):
        """Test logger name hierarchy."""
        parent = setup_logger('test_parent')
        child = setup_logger('test_parent.child')
        
        assert parent.name == 'test_parent'
        assert child.name == 'test_parent.child'
    
    @pytest.mark.integration
    def test_full_workflow(self):
        """Test complete logger workflow."""
        # Setup
        logger = setup_logger('test_workflow', level=logging.INFO)
        assert logger.level == logging.INFO
        
        # Log at various levels
        logger.debug("Debug test")
        logger.info("Info test")
        logger.warning("Warning test")


class TestLoggerFormatting:
    """Test logger formatting."""
    
    @pytest.mark.unit
    def test_logger_has_formatter(self):
        """Test logger handlers have formatters."""
        logger = setup_logger('test_formatter')
        
        for handler in logger.handlers:
            assert handler.formatter is not None
    
    @pytest.mark.unit
    def test_formatter_includes_timestamp(self):
        """Test formatter includes timestamp."""
        logger = setup_logger('test_timestamp')
        
        formatter = logger.handlers[0].formatter
        format_string = formatter._fmt
        
        assert '%(asctime)s' in format_string
    
    @pytest.mark.unit
    def test_formatter_includes_name(self):
        """Test formatter includes logger name."""
        logger = setup_logger('test_name_format')
        
        formatter = logger.handlers[0].formatter
        format_string = formatter._fmt
        
        assert '%(name)s' in format_string
    
    @pytest.mark.unit
    def test_formatter_includes_level(self):
        """Test formatter includes log level."""
        logger = setup_logger('test_level_format')
        
        formatter = logger.handlers[0].formatter
        format_string = formatter._fmt
        
        assert '%(levelname)s' in format_string
    
    @pytest.mark.unit
    def test_formatter_includes_message(self):
        """Test formatter includes message."""
        logger = setup_logger('test_message_format')
        
        formatter = logger.handlers[0].formatter
        format_string = formatter._fmt
        
        assert '%(message)s' in format_string


class TestLoggerEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.edge_case
    def test_setup_logger_empty_name(self):
        """Test setup_logger with empty name."""
        logger = setup_logger('')
        
        assert logger is not None
        # Python logging uses 'root' for empty string names
        assert logger.name in ('', 'root')
    
    @pytest.mark.edge_case
    def test_setup_logger_special_characters(self):
        """Test logger name with special characters."""
        logger = setup_logger('test.logger-with_special/chars')
        
        assert logger is not None
        assert logger.name == 'test.logger-with_special/chars'