"""
Tests for the crisis detection service
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

from services.crisis_detector import CrisisDetectionService, CrisisDetectionResult
from utils.exceptions import CrisisDetectionException


class TestCrisisDetectionService:
    """Test cases for CrisisDetectionService"""
    
    @pytest.fixture
    def service(self):
        """Create a crisis detection service instance"""
        return CrisisDetectionService()
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return {
            'crisis': [
                "I want to kill myself",
                "I can't go on anymore",
                "I feel hopeless and worthless",
                "I'm planning to end it all tonight"
            ],
            'non_crisis': [
                "I'm feeling great today!",
                "Life is wonderful",
                "I love my family",
                "Everything is going well"
            ]
        }
    
    @pytest.mark.asyncio
    async def test_initialize(self, service):
        """Test service initialization"""
        with patch.object(service, '_load_models') as mock_load:
            with patch.object(service, '_initialize_ensemble') as mock_ensemble:
                await service.initialize()
                
                assert service.is_initialized
                mock_load.assert_called_once()
                mock_ensemble.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, service):
        """Test initialization failure"""
        with patch.object(service, '_load_models', side_effect=Exception("Model load failed")):
            with pytest.raises(CrisisDetectionException):
                await service.initialize()
    
    @pytest.mark.asyncio
    async def test_analyze_text_crisis(self, service, sample_texts):
        """Test crisis text analysis"""
        # Mock the service as initialized
        service.is_initialized = True
        service.models = {'ensemble': Mock()}
        service.ensemble = Mock()
        
        # Mock preprocessing
        with patch.object(service.preprocessor, 'preprocess') as mock_preprocess:
            mock_preprocess.return_value = {
                'processed_text': 'processed text',
                'features': {}
            }
            
            # Mock feature extraction
            with patch.object(service.feature_extractor, 'extract_features') as mock_extract:
                mock_extract.return_value = {'crisis_score': 0.8}
                
                # Mock model prediction
                with patch.object(service, '_predict_sklearn') as mock_predict:
                    mock_predict.return_value = (1, 0.8)  # crisis prediction with high probability
                    
                    result = await service.analyze_text(sample_texts['crisis'][0])
                    
                    assert isinstance(result, CrisisDetectionResult)
                    assert result.crisis_probability > 0.5
                    assert result.risk_level in ['high', 'critical']
                    assert result.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_analyze_text_non_crisis(self, service, sample_texts):
        """Test non-crisis text analysis"""
        service.is_initialized = True
        service.models = {'ensemble': Mock()}
        service.ensemble = Mock()
        
        with patch.object(service.preprocessor, 'preprocess') as mock_preprocess:
            mock_preprocess.return_value = {
                'processed_text': 'processed text',
                'features': {}
            }
            
            with patch.object(service.feature_extractor, 'extract_features') as mock_extract:
                mock_extract.return_value = {'crisis_score': 0.1}
                
                with patch.object(service, '_predict_sklearn') as mock_predict:
                    mock_predict.return_value = (0, 0.1)  # non-crisis prediction
                    
                    result = await service.analyze_text(sample_texts['non_crisis'][0])
                    
                    assert isinstance(result, CrisisDetectionResult)
                    assert result.crisis_probability < 0.5
                    assert result.risk_level in ['low', 'medium']
    
    @pytest.mark.asyncio
    async def test_analyze_text_not_initialized(self, service):
        """Test analysis when service is not initialized"""
        with pytest.raises(CrisisDetectionException):
            await service.analyze_text("test text")
    
    @pytest.mark.asyncio
    async def test_analyze_batch(self, service, sample_texts):
        """Test batch analysis"""
        service.is_initialized = True
        service.models = {'ensemble': Mock()}
        service.ensemble = Mock()
        
        texts = sample_texts['crisis'] + sample_texts['non_crisis']
        
        with patch.object(service, 'analyze_text') as mock_analyze:
            # Mock individual analysis results
            mock_results = [
                CrisisDetectionResult(0.8, 'high', 0.9, 'ensemble'),
                CrisisDetectionResult(0.7, 'high', 0.8, 'ensemble'),
                CrisisDetectionResult(0.2, 'low', 0.7, 'ensemble'),
                CrisisDetectionResult(0.1, 'low', 0.6, 'ensemble')
            ]
            mock_analyze.side_effect = mock_results
            
            results = await service.analyze_batch(texts)
            
            assert len(results) == len(texts)
            assert all(isinstance(r, CrisisDetectionResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_analyze_batch_with_exceptions(self, service):
        """Test batch analysis with exceptions"""
        service.is_initialized = True
        
        texts = ["text1", "text2", "text3"]
        
        with patch.object(service, 'analyze_text') as mock_analyze:
            # First call succeeds, second fails, third succeeds
            mock_analyze.side_effect = [
                CrisisDetectionResult(0.5, 'medium', 0.7, 'ensemble'),
                Exception("Analysis failed"),
                CrisisDetectionResult(0.3, 'low', 0.6, 'ensemble')
            ]
            
            results = await service.analyze_batch(texts)
            
            assert len(results) == 3
            assert isinstance(results[0], CrisisDetectionResult)
            assert isinstance(results[1], CrisisDetectionResult)  # Fallback result
            assert isinstance(results[2], CrisisDetectionResult)
    
    def test_determine_risk_level(self, service):
        """Test risk level determination"""
        # Test different probability thresholds
        assert service._determine_risk_level(0.9) == "critical"
        assert service._determine_risk_level(0.8) == "high"
        assert service._determine_risk_level(0.6) == "medium"
        assert service._determine_risk_level(0.3) == "low"
        
        # Test with custom threshold
        assert service._determine_risk_level(0.7, threshold=0.8) == "low"
        assert service._determine_risk_level(0.9, threshold=0.8) == "critical"
    
    def test_calculate_confidence_score(self, service):
        """Test confidence score calculation"""
        # Test with high agreement
        probabilities = {'model1': 0.8, 'model2': 0.82, 'model3': 0.79}
        confidence = service._calculate_confidence_score(probabilities, 0.8)
        assert confidence > 0.8  # High confidence due to agreement
        
        # Test with low agreement
        probabilities = {'model1': 0.2, 'model2': 0.8, 'model3': 0.5}
        confidence = service._calculate_confidence_score(probabilities, 0.5)
        assert confidence < 0.5  # Low confidence due to disagreement
    
    def test_rule_based_detection(self, service):
        """Test rule-based crisis detection"""
        # Test crisis text
        crisis_text = "I want to kill myself and end it all"
        score = service._rule_based_detection(crisis_text)
        assert score > 0.5
        
        # Test non-crisis text
        non_crisis_text = "I'm feeling great and happy today"
        score = service._rule_based_detection(non_crisis_text)
        assert score < 0.3
    
    def test_extract_key_features(self, service):
        """Test key feature extraction"""
        text = "I feel hopeless and want to die"
        features = service._extract_key_features(text, {})
        
        assert 'crisis_keywords' in features
        assert 'sentiment_scores' in features
        assert 'text_length' in features
        assert 'word_count' in features
        assert len(features['crisis_keywords']) > 0
    
    @pytest.mark.asyncio
    async def test_generate_explanation(self, service):
        """Test explanation generation"""
        text = "I feel hopeless"
        crisis_prob = 0.8
        
        with patch('lime.lime_text.LimeTextExplainer') as mock_lime:
            mock_explainer = Mock()
            mock_lime.return_value = mock_explainer
            mock_explainer.explain_instance.return_value.as_list.return_value = [
                ('hopeless', 0.5),
                ('feel', 0.3)
            ]
            
            explanation = await service._generate_explanation(text, crisis_prob)
            
            assert 'lime_explanation' in explanation
            assert 'decision_factors' in explanation
            assert len(explanation['decision_factors']) > 0
    
    def test_is_ready(self, service):
        """Test service readiness check"""
        # Not initialized
        assert not service.is_ready()
        
        # Initialized but no models
        service.is_initialized = True
        service.models = {}
        assert not service.is_ready()
        
        # Initialized with models
        service.models = {'ensemble': Mock()}
        assert service.is_ready()
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, service):
        """Test model information retrieval"""
        service.models = {
            'bert': {'name': 'bert-base-uncased', 'type': 'transformer'},
            'ensemble': {'type': 'sklearn'}
        }
        
        info = await service.get_model_info()
        
        assert 'bert' in info
        assert 'ensemble' in info
        assert info['bert']['type'] == 'transformer'
        assert info['ensemble']['type'] == 'sklearn'


class TestCrisisDetectionResult:
    """Test cases for CrisisDetectionResult"""
    
    def test_crisis_detection_result_creation(self):
        """Test CrisisDetectionResult creation"""
        result = CrisisDetectionResult(
            crisis_probability=0.8,
            risk_level='high',
            confidence_score=0.9,
            model_name='ensemble',
            key_features={'crisis_keywords': ['hopeless']},
            explanation={'lime_explanation': []},
            processing_time=0.5
        )
        
        assert result.crisis_probability == 0.8
        assert result.risk_level == 'high'
        assert result.confidence_score == 0.9
        assert result.model_name == 'ensemble'
        assert result.key_features == {'crisis_keywords': ['hopeless']}
        assert result.explanation == {'lime_explanation': []}
        assert result.processing_time == 0.5
    
    def test_crisis_detection_result_defaults(self):
        """Test CrisisDetectionResult with defaults"""
        result = CrisisDetectionResult(
            crisis_probability=0.5,
            risk_level='medium',
            confidence_score=0.7,
            model_name='test'
        )
        
        assert result.key_features == {}
        assert result.explanation == {}
        assert result.processing_time == 0.0


@pytest.mark.integration
class TestCrisisDetectionIntegration:
    """Integration tests for crisis detection"""
    
    @pytest.fixture
    async def initialized_service(self):
        """Create and initialize a service for integration tests"""
        service = CrisisDetectionService()
        
        # Mock the initialization to avoid actual model loading
        with patch.object(service, '_load_models'):
            with patch.object(service, '_initialize_ensemble'):
                await service.initialize()
        
        return service
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis(self, initialized_service):
        """Test end-to-end analysis pipeline"""
        crisis_text = "I want to kill myself and end it all"
        
        with patch.object(initialized_service.preprocessor, 'preprocess') as mock_preprocess:
            mock_preprocess.return_value = {
                'processed_text': crisis_text,
                'features': {'crisis_score': 0.8}
            }
            
            with patch.object(initialized_service.feature_extractor, 'extract_features') as mock_extract:
                mock_extract.return_value = {'crisis_score': 0.8}
                
                with patch.object(initialized_service, '_predict_sklearn') as mock_predict:
                    mock_predict.return_value = (1, 0.8)
                    
                    result = await initialized_service.analyze_text(crisis_text)
                    
                    assert result.crisis_probability > 0.5
                    assert result.risk_level in ['high', 'critical']
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, initialized_service):
        """Test batch processing performance"""
        texts = [f"Test text {i}" for i in range(100)]
        
        with patch.object(initialized_service, 'analyze_text') as mock_analyze:
            mock_analyze.return_value = CrisisDetectionResult(
                0.5, 'medium', 0.7, 'ensemble'
            )
            
            start_time = asyncio.get_event_loop().time()
            results = await initialized_service.analyze_batch(texts)
            end_time = asyncio.get_event_loop().time()
            
            assert len(results) == 100
            assert (end_time - start_time) < 10.0  # Should complete within 10 seconds


@pytest.mark.performance
class TestCrisisDetectionPerformance:
    """Performance tests for crisis detection"""
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing large batches of text"""
        service = CrisisDetectionService()
        service.is_initialized = True
        service.models = {'ensemble': Mock()}
        
        # Create large batch
        texts = [f"Test text {i} with some content" for i in range(1000)]
        
        with patch.object(service, 'analyze_text') as mock_analyze:
            mock_analyze.return_value = CrisisDetectionResult(
                0.5, 'medium', 0.7, 'ensemble'
            )
            
            start_time = asyncio.get_event_loop().time()
            results = await service.analyze_batch(texts)
            end_time = asyncio.get_event_loop().time()
            
            assert len(results) == 1000
            processing_time = end_time - start_time
            assert processing_time < 30.0  # Should complete within 30 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        service = CrisisDetectionService()
        service.is_initialized = True
        service.models = {'ensemble': Mock()}
        
        # Process large batch
        texts = [f"Test text {i}" for i in range(500)]
        
        with patch.object(service, 'analyze_text') as mock_analyze:
            mock_analyze.return_value = CrisisDetectionResult(
                0.5, 'medium', 0.7, 'ensemble'
            )
            
            results = await service.analyze_batch(texts)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024
